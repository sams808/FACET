"""The Model window: an MD model opened, set up, analysed and exported.

A window of its own, one per model, beside the crystal window rather than a
tab in it: the crystal window's panels follow its active ``Structure`` and
would each compute on a 10 000-atom frame handed to them (minutes, measured
by the UI scouting of 2026-10-07), its tab column is too narrow for a
histogram beside a table, and a separate window can sit next to a reference
crystal. It has no Qt parent, so it never changes what the crystal window's
``findChild`` finds; :func:`open_model_window` keeps the reference that keeps
it alive.

The flow:

1. **Reading** (a worker thread, :class:`~.md_jobs.Job` running
   :func:`~.md_jobs.open_model`). A file the reader refuses for want of an
   input only the user has (a type map, a topology, a box, units) opens the
   :class:`~.md_dialogs.ReadOptionsPanel`, which shows the reader's message
   and asks for exactly those inputs; any other refusal is shown in full with
   a way to open another file. ``trajectory`` is None until the file reads.
2. **Setup** (:class:`~.md_dialogs.SetupPanel`): oxidation states, network
   formers (none ticked), frames, thresholds, and the analysis checklist with
   every option group of ``md_analysis.AnalysisRequest``.
3. **Run** (:class:`~.md_jobs.AnalysisJob`): ``md_analysis.analyse`` on a
   worker thread; the status bar shows the engine's stage, a progress bar
   and Cancel. A cancelled run shows what was finished, with its note.
4. **Results**: the result browser (``md_views.ResultBrowser`` when that
   module is present, else a list of every table with its rows), the
   threshold panel (``md_threshold.ThresholdPanel``), the 3D view of one
   frame (``md_scene.build_md_scene`` from the bulk engine's bonds, drawn by
   the existing ``StructureView``; on the QPainter tier above
   :data:`ATOMS_ONLY_ABOVE` atoms the atoms are drawn without bonds, with a
   note), and every note and provenance line of the run.
5. **Export**: File > Export writes ``md_export``'s CSV directory or XLSX
   workbook on a worker thread, each table with its provenance header.

Closing the window cancels its work and waits at most
:data:`CLOSE_WAIT_MS` for it; work that takes longer to notice the cancel
(the engine looks once per frame) finishes in the background, held by
``md_jobs`` so that no running thread is destroyed.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import theme as theme_mod
from ..version import NAME
from . import chrome, md_jobs
from .md_dialogs import ReadOptionsPanel, SetupPanel, _escape, _hint

__all__ = ["ModelWindow", "open_model_window", "model_windows",
           "md_file_filter", "ATOMS_ONLY_ABOVE", "CLOSE_WAIT_MS"]

# Above this many atoms the QPainter tier (no OpenGL) draws the atoms of a
# frame without bonds. A choice from the UI scouting's measurements of
# 2026-10-07: the QPainter tier took 0.64 s per frame at 3 000 atoms and
# 1.9-6.4 s at 9 000-24 000, bonds being 69 % of it.
ATOMS_ONLY_ABOVE = 5000

# How long a closing window waits for its work to stop before leaving it to
# finish in the background. A choice: long enough for a cancel between two
# frames of a small model, short enough that closing never feels stuck.
CLOSE_WAIT_MS = 3000

# How many rows of one table the fallback result list shows.
_ROWS_SHOWN = 2000

_WINDOWS: list = []


def model_windows() -> tuple:
    """Every Model window open in this process."""
    return tuple(_WINDOWS)


def md_file_filter() -> str:
    """A file-dialog filter of the MD formats read_trajectory reads: the
    crystal window's own (``preview.md_file_filter``), so both File > Open
    MD model… list the same names; a list of the names when that window is
    not in this build."""
    try:
        from .preview import md_file_filter as crystal_window_filter

        return crystal_window_filter()
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


def _alive(widget) -> bool:
    """Whether a Qt object still exists (its window may have closed)."""
    if widget is None:
        return False
    try:
        import shiboken6

        return bool(shiboken6.isValid(widget))
    except Exception:              # noqa: BLE001 - not a Qt object
        return True


def _normalise_source(paths):
    if isinstance(paths, (list, tuple)):
        items = [str(p) for p in paths]
        return items[0] if len(items) == 1 else items
    return str(paths)


# ---------------------------------------------------------------------------
# fallback views, used while md_views / md_threshold are not present
# ---------------------------------------------------------------------------

class _ResultList(QWidget):
    """Every table of a run, and the rows of the one selected.

    Stands in for ``md_views.ResultBrowser`` when that module is not
    present: the descriptors by analysis, an analysis that produced nothing
    with its reason, and the rows ``md_export`` would write.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["analysis / descriptor", "kind", "rows"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.currentItemChanged.connect(self._on_item)
        self.table = QTableWidget()
        self.table.verticalHeader().setVisible(False)
        self.note = _hint()
        split = QSplitter(Qt.Vertical)
        split.addWidget(self.tree)
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self.note)
        column.addWidget(self.table)
        split.addWidget(holder)
        split.setSizes([300, 400])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(split)

    def set_result(self, result) -> None:
        self.result = result
        self.tree.clear()
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        for name, output in result.outputs.items():
            top = QTreeWidgetItem([name, "", ""])
            top.setData(0, Qt.UserRole, None)
            if output.error:
                child = QTreeWidgetItem([f"not computed: {output.error}",
                                         "", ""])
                child.setToolTip(0, output.error)
                top.addChild(child)
            for key, container in output.tables.items():
                kind = type(container).__name__
                try:
                    from ..core import md_export

                    rows = len(md_export.descriptor_rows(container))
                except ValueError:
                    rows = 0
                child = QTreeWidgetItem([key, kind, str(rows)])
                child.setData(0, Qt.UserRole, (name, key))
                top.addChild(child)
            self.tree.addTopLevelItem(top)
        self.tree.expandToDepth(0)
        self.note.setText("Select a descriptor to see its rows.")

    def tables(self) -> list[tuple[str, str]]:
        out = []
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            for j in range(top.childCount()):
                data = top.child(j).data(0, Qt.UserRole)
                if data:
                    out.append(tuple(data))
        return out

    @Slot(QTreeWidgetItem, QTreeWidgetItem)
    def _on_item(self, item, _previous) -> None:
        from ..core import md_export

        if item is None or self.result is None:
            return
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        name, key = data
        container = self.result.outputs[name].tables[key]
        try:
            rows = md_export.descriptor_rows(container)
        except ValueError as error:
            self.note.setText(str(error))
            return
        columns = list(rows[0]) if rows else []
        shown = rows[:_ROWS_SHOWN]
        self.table.clear()
        self.table.setColumnCount(len(columns))
        self.table.setRowCount(len(shown))
        self.table.setHorizontalHeaderLabels(columns)
        for r, row in enumerate(shown):
            for c, column in enumerate(columns):
                value = row.get(column)
                text = "" if value is None else (
                    f"{value:.6g}" if isinstance(value, float) else str(value))
                self.table.setItem(r, c, QTableWidgetItem(text))
        notes = list(getattr(container, "notes", ()) or ())
        more = (f" (the first {_ROWS_SHOWN} of {len(rows)} rows; the "
                "export holds them all)") if len(rows) > _ROWS_SHOWN else ""
        self.note.setText(f"{name}: {key}{more}"
                          + ("\n" + "\n".join(notes) if notes else ""))


# ---------------------------------------------------------------------------
# the 3D view of one frame
# ---------------------------------------------------------------------------

class _FrameTab(QWidget):
    """One frame of the model in the existing StructureView."""

    showRequested = Signal(int)

    def __init__(self, n_frames: int, n_atoms: int, parent=None, *,
                 theme=None):
        super().__init__(parent)
        from ..gl.view import StructureView

        self.n_atoms = int(n_atoms)
        self.frame_box = QSpinBox()
        self.frame_box.setRange(0, max(0, int(n_frames) - 1))
        chrome.name_inside(self.frame_box, "frame")
        self.show_button = QPushButton("&Show frame")
        self.show_button.clicked.connect(self._on_show)
        self.note = _hint()
        row = QHBoxLayout()
        row.addWidget(self.frame_box)
        row.addWidget(self.show_button)
        row.addWidget(self.note, 1)
        self.view = StructureView()
        if theme is not None:
            self.view.set_theme(theme)
        column = QVBoxLayout(self)
        column.setContentsMargins(4, 4, 4, 4)
        column.addLayout(row)
        column.addWidget(self.view, 1)

    def painter_tier(self) -> bool:
        """True on the QPainter tier, or while the view has not said."""
        from ..gl import caps as caps_mod

        caps = self.view.caps
        return caps is None or caps.tier is caps_mod.Tier.BASIC \
            or self.view._fallback is not None

    def with_bonds(self) -> bool:
        return not (self.painter_tier() and self.n_atoms > ATOMS_ONLY_ABOVE)

    @Slot()
    def _on_show(self) -> None:
        self.showRequested.emit(int(self.frame_box.value()))


# ---------------------------------------------------------------------------
# the window
# ---------------------------------------------------------------------------

class ModelWindow(QMainWindow):
    """One MD model: reading, setup, run, results, export.

    ``paths``: one path (a file, a directory or a wildcard pattern) or a
    list of files read as one trajectory. ``read_options`` are passed to
    ``md_readers.read_trajectory``. Attributes: ``trajectory`` (None while
    the file is being read or the reader asks for an input), ``summary``,
    ``request`` (the setup's AnalysisRequest, None while it cannot run),
    ``result`` (the last ModelResult, or None). ``finished(object)`` is
    emitted after each run with the ModelResult, or with a
    :class:`~.md_jobs.JobFailure` when the run produced nothing.
    """

    finished = Signal(object)
    opened = Signal(object)          # ModelSummary, or ReadProblem
    closed = Signal(object)

    def __init__(self, paths, parent=None, *, read_options=None):
        # No Qt parent: see the module docstring.
        super().__init__(None)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self._opener = parent
        theme = getattr(parent, "theme", None)
        self.theme = theme if theme is not None else theme_mod.Theme()
        self.source = _normalise_source(paths)
        self.read_options = dict(read_options or {})
        self.trajectory = None
        self.summary = None
        self.problem = None
        self.result = None
        # the last run that finished uncancelled, and a cancelled run's
        # partial result since: a cancel never takes a complete run away
        self.complete_result = None
        self.partial_result = None
        self.companions: tuple = ()
        self._manual = None
        self.last_failure = None
        self.last_request = None
        self.setup: SetupPanel | None = None
        self.results_view = None
        self.threshold_panel = None
        self.frame_tab: _FrameTab | None = None
        self.frame_view = None
        self._open_job = None
        self._run_job = None
        self._frame_job = None
        self._export_job = None
        self._links: dict = {}
        self._closing = False
        self._pending_frame = None
        self._model_generation = 0

        self.setWindowTitle(f"{md_jobs.source_label(self.source)} - "
                            f"{NAME} Model")
        self.resize(1440, 920)
        self.setAcceptDrops(True)

        self.pages = QStackedWidget()
        self.setCentralWidget(self.pages)
        self.reading_page = self._build_reading_page()
        self.read_panel = ReadOptionsPanel(theme=self.theme)
        self.read_panel.setMaximumWidth(1100)
        self.read_panel.readRequested.connect(self._on_read_again)
        self.read_panel.openOtherRequested.connect(self.choose_md_file)
        self.read_page = chrome.in_scroll_area(self.read_panel)
        self.workspace = QWidget()
        self.pages.addWidget(self.reading_page)
        self.pages.addWidget(self.read_page)
        self.pages.addWidget(self.workspace)
        self._build_status()
        self._build_menu()
        self.open_model()

    # -- building ------------------------------------------------------------
    def _build_reading_page(self) -> QWidget:
        page = QWidget()
        column = QVBoxLayout(page)
        column.addStretch(1)
        self.reading_label = QLabel()
        self.reading_label.setAlignment(Qt.AlignCenter)
        self.reading_label.setWordWrap(True)
        busy = QProgressBar()
        busy.setRange(0, 0)
        busy.setMaximumWidth(360)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(busy)
        row.addStretch(1)
        column.addWidget(self.reading_label)
        column.addLayout(row)
        column.addStretch(2)
        return page

    def _build_status(self) -> None:
        bar = self.statusBar()
        self.stage_label = QLabel()
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(320)
        self.progress.setTextVisible(True)
        self.cancel_button = QPushButton("Cancel run")
        self.cancel_button.clicked.connect(self.cancel)
        bar.addWidget(self.stage_label, 1)
        bar.addPermanentWidget(self.progress)
        bar.addPermanentWidget(self.cancel_button)
        self._set_running(False)

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("&File")
        self.open_action = QAction("&Open MD model…", self)
        self.open_action.setShortcut(QKeySequence.Open)
        self.open_action.triggered.connect(self.choose_md_file)
        self.export_csv_action = QAction("Export results as CSV &files…",
                                         self)
        self.export_csv_action.triggered.connect(self._choose_export_csv)
        self.export_xlsx_action = QAction("Export results as &XLSX "
                                          "workbook…", self)
        self.export_xlsx_action.triggered.connect(self._choose_export_xlsx)
        # bound slots, not lambdas: a lambda holding self in a connection
        # kept every closed window (844 widgets, 27 MB each) alive
        self.export_figure_action = QAction("Save the fi&gure shown…", self)
        self.export_figure_action.triggered.connect(self._save_figure_shown)
        self.export_rows_action = QAction("Save the ro&ws shown…", self)
        self.export_rows_action.triggered.connect(self._save_rows_shown)
        self.save_request_action = QAction(
            "Save the re&quest for the command line…", self)
        self.save_request_action.setToolTip(
            "A request file (TOML) holding the setup: py -3.11 -m facet.md "
            "analyse FILE --request it.toml repeats the run without the "
            "window.")
        self.save_request_action.triggered.connect(self._choose_save_request)
        self.close_action = QAction("&Close", self)
        self.close_action.setShortcut(QKeySequence.Close)
        self.close_action.triggered.connect(self.close)
        menu.addAction(self.open_action)
        menu.addSeparator()
        menu.addAction(self.export_csv_action)
        menu.addAction(self.export_xlsx_action)
        menu.addAction(self.export_figure_action)
        menu.addAction(self.export_rows_action)
        menu.addSeparator()
        menu.addAction(self.save_request_action)
        menu.addSeparator()
        menu.addAction(self.close_action)

        run = self.menuBar().addMenu("&Run")
        self.run_action = QAction("&Run analyses", self)
        self.run_action.setShortcut(QKeySequence("Ctrl+R"))
        self.run_action.triggered.connect(self._run_from_menu)
        self.cancel_action = QAction("&Cancel run", self)
        self.cancel_action.triggered.connect(self.cancel)
        self.show_complete_action = QAction("Show the &last complete run",
                                            self)
        self.show_complete_action.setToolTip(
            "The last run that finished uncancelled; a cancelled run "
            "after it does not replace it.")
        self.show_complete_action.triggered.connect(self.show_complete_result)
        self.show_partial_action = QAction(
            "Show the cancelled run's &partial result", self)
        self.show_partial_action.triggered.connect(self.show_partial_result)
        run.addAction(self.run_action)
        run.addAction(self.cancel_action)
        run.addSeparator()
        run.addAction(self.show_complete_action)
        run.addAction(self.show_partial_action)

        help_menu = self.menuBar().addMenu("&Help")
        self.help_action = QAction("MD models and the Model &window", self)
        self.help_action.setShortcut(QKeySequence.HelpContents)
        self.help_action.setToolTip("The manual's MD section: type maps, "
                                    "formers, frames, every analysis and "
                                    "what it needs, the threshold, export.")
        self.help_action.triggered.connect(self.show_manual)
        help_menu.addAction(self.help_action)
        self._update_actions()

    def _build_workspace(self) -> None:
        """The setup on the left, the results on the right."""
        old = self.workspace
        self.workspace = QWidget()
        self.pages.insertWidget(2, self.workspace)
        self.pages.removeWidget(old)
        old.deleteLater()
        layout = QVBoxLayout(self.workspace)
        layout.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Horizontal)
        self.setup = SetupPanel(self.summary, self.trajectory,
                                theme=self.theme,
                                type_map=self.summary.type_map)
        self.setup.runRequested.connect(self._run_from_menu)
        self.setup.changed.connect(self._update_actions)
        self.setup.rereadRequested.connect(self.reread)
        scroll = chrome.in_scroll_area(self.setup)
        # the Run button and what the run needs, under the scroll area: in
        # view whatever part of the (long) setup is scrolled to
        left = QWidget()
        left_column = QVBoxLayout(left)
        left_column.setContentsMargins(0, 0, 0, 0)
        left_column.setSpacing(0)
        left_column.addWidget(scroll, 1)
        footer = self.setup.detach_footer()
        footer.setContentsMargins(8, 4, 8, 6)
        left_column.addWidget(footer)
        left.setMinimumWidth(420)
        self.setup_scroll = scroll
        split.addWidget(left)

        self.tabs = QTabWidget()
        self.results_view = self._make_results_view()
        self._set_empty_text(running=False)
        self.tabs.addTab(self.results_view, "Results")
        self.threshold_panel = self._make_threshold_panel()
        self.tabs.addTab(self.threshold_panel, "Threshold")
        self.frame_tab = _FrameTab(self.summary.n_frames,
                                   self.summary.n_atoms, theme=self.theme)
        self.frame_tab.showRequested.connect(self.show_frame)
        self.frame_tab.view.ready.connect(self._on_view_ready)
        self.tabs.addTab(self.frame_tab, "3D view")
        self.notes_text = QPlainTextEdit()
        self.notes_text.setReadOnly(True)
        self.notes_text.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.tabs.addTab(self.notes_text, "Notes and provenance")
        self._show_model_notes()
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        # the results take what the setup's fields leave: the browser's tree
        # with its mean ± std, and a figure with its rows under it
        split.setSizes([500, 940])
        layout.addWidget(split)
        self.splitter = split

    # What the result browser says while it holds no run.
    EMPTY_TEXT = ("No run yet. Tick analyses in the setup on the left and "
                  "press Run analyses (Ctrl+R); the results are listed here.")
    RUNNING_TEXT = ("A run is in progress (its stage and Cancel run are in "
                    "the status bar below); its results are listed here when "
                    "it ends.")

    def _set_empty_text(self, running: bool) -> None:
        empty = getattr(self.results_view, "set_empty_text", None)
        if empty is not None:
            empty(self.RUNNING_TEXT if running else self.EMPTY_TEXT)

    def _make_results_view(self):
        try:
            from .md_views import ResultBrowser
        except ImportError:
            ResultBrowser = None
        if ResultBrowser is not None:
            try:
                try:
                    browser = ResultBrowser(theme=self.theme)
                except TypeError:
                    browser = ResultBrowser()
            except Exception as error:     # noqa: BLE001 - shown, not raised
                fallback = _ResultList()
                fallback.note.setText(f"md_views.ResultBrowser could not be "
                                      f"built ({error}); the tables are "
                                      "listed here.")
                return fallback
            status = getattr(browser, "statusMessage", None)
            if status is not None:
                status.connect(self._on_browser_status)
            # File > Save the figure / rows shown follow the browser's own
            # actions, which know whether the item shown has a figure
            for name in ("export_figure_action", "export_rows_action"):
                action = getattr(browser, name, None)
                if action is not None:
                    action.changed.connect(self._update_actions)
            return browser
        view = _ResultList()
        view.note.setText("No run yet: tick analyses on the left and press "
                          "Run. (md_views is not present in this build, so "
                          "the tables are listed here without plots.)")
        return view

    def _make_threshold_panel(self):
        try:
            from .md_threshold import ThresholdPanel
        except ImportError:
            ThresholdPanel = None
        if ThresholdPanel is not None:
            try:
                try:
                    panel = ThresholdPanel(theme=self.theme)
                except TypeError:
                    panel = ThresholdPanel()
            except Exception as error:     # noqa: BLE001 - shown, not raised
                return _hint(f"md_threshold.ThresholdPanel could not be "
                             f"built: {error}")
            changed = getattr(panel, "thresholdChanged", None)
            if changed is not None:
                changed.connect(self._on_panel_threshold)
            status = getattr(panel, "statusMessage", None)
            if status is not None:
                status.connect(self._on_browser_status)
            return panel
        label = _hint("The threshold panel (facet.ui.md_threshold) is not "
                      "present in this build.")
        label.setAlignment(Qt.AlignCenter)
        return label

    # -- reading -------------------------------------------------------------
    def open_model(self, read_options=None) -> None:
        """Read the source (again) on a worker thread."""
        if self._open_job is not None and self._open_job.pending:
            self.stage_label.setText("The model file is being read")
            return
        if read_options is not None:
            self.read_options = dict(read_options)
        source = self.source
        options = dict(self.read_options)
        name = _escape(md_jobs.source_label(source))
        self.reading_label.setText(f"Reading {name} …")
        self.pages.setCurrentWidget(self.reading_page)
        self.stage_label.setText("Reading the model file")

        def work(progress, cancelled):
            return md_jobs.open_model(source, options)

        job = md_jobs.Job(work, name="open")
        self._connect(job, succeeded=self._on_open_done,
                      failed=self._on_open_failed)
        self._open_job = job
        job.start()

    @Slot(object)
    def _on_open_done(self, outcome) -> None:
        if self._closing:
            return
        if isinstance(outcome, md_jobs.Opened):
            previous = self.trajectory
            self.trajectory = outcome.trajectory
            self.summary = outcome.summary
            self.problem = None
            # a model read again (other read options) is a new model: its
            # results and frame are those of the new read, and a frame
            # still loading from the old read is not shown
            self.result = self.complete_result = self.partial_result = None
            self.frame_view = None
            self._model_generation += 1
            if previous is not None and previous is not outcome.trajectory \
                    and not self._busy():
                close = getattr(previous, "close", None)
                if close is not None:
                    close()
            self._build_workspace()
            self.pages.setCurrentWidget(self.workspace)
            self.stage_label.setText(
                f"Read {self.summary.label}: {self.summary.n_atoms} atoms × "
                f"{self.summary.n_frames} frame(s) ({self.summary.file_format})")
            self._update_actions()
            self.opened.emit(self.summary)
            QTimer.singleShot(0, self._first_frame)
            return
        self.trajectory = None
        self.problem = outcome
        if self.companions and hasattr(self.read_panel, "set_companions"):
            self.read_panel.companions = tuple(self.companions)
        self.read_panel.set_problem(outcome)
        self.pages.setCurrentWidget(self.read_page)
        self.stage_label.setText(
            "The reader asks for more information" if outcome.can_supply
            else "The file was not read")
        self._update_actions()
        self.opened.emit(outcome)

    @Slot(object)
    def _on_open_failed(self, failure) -> None:
        """An exception the read did not sort into a refusal: shown in
        full, and the fields of the last refusal stay (the options it asked
        for, the options the format takes), so the inputs can be typed
        again rather than the window dead-ending."""
        if self._closing:
            return
        previous = self.problem
        if previous is not None:
            # the options of the read that failed, so its fields show what
            # was tried (and Read again waits for a change to them)
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
            self.stage_label.setText("A run is in progress: cancel it, or "
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
        setter = getattr(self.read_panel, "set_companions", None)
        if setter is not None:
            setter(self.companions)

    @Slot()
    def choose_md_file(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Open MD model", "",
                                                md_file_filter())
        if paths:
            self.open_paths(paths if len(paths) > 1 else paths[0])

    def open_paths(self, paths) -> "ModelWindow":
        """Open ``paths`` here when this window holds no model yet, else in a
        new Model window; returns the window that reads them."""
        if self.trajectory is None and not self._busy():
            self.source = _normalise_source(paths)
            self.read_options = {}
            self.companions = ()
            self.read_panel.companions = ()
            self.setWindowTitle(f"{md_jobs.source_label(self.source)} - "
                                f"{NAME} Model")
            self.open_model({})
            return self
        return open_model_window(paths, self._opener)

    # -- the run -------------------------------------------------------------
    @property
    def request(self):
        """The setup's AnalysisRequest (None while it cannot run)."""
        if self.setup is None:
            return None
        return self.setup.request()

    def run_analysis(self, request=None) -> bool:
        """Start a run of ``request`` (the setup's when None); False when
        nothing was started, with the reason in the status bar."""
        if self.trajectory is None:
            self.stage_label.setText("No model is read yet")
            return False
        if self._run_job is not None and self._run_job.pending:
            self.stage_label.setText("A run is in progress")
            return False
        if request is None or isinstance(request, bool):
            request = self.request
        if request is None:
            problems = self.setup.problems() if self.setup else []
            self.stage_label.setText("Run needs: " + "; ".join(problems))
            return False
        self.last_request = request
        job = md_jobs.AnalysisJob(self.trajectory, request)
        self._connect(job, progressed=self._on_run_progress,
                      succeeded=self._on_run_finished,
                      failed=self._on_run_failed, ended=self._on_run_ended)
        self._run_job = job
        job.start()
        self._set_running(True)
        self.progress.setRange(0, 0)
        self.stage_label.setText("Starting: " + ", ".join(request.analyses))
        return True

    def cancel(self) -> None:
        """Ask the run to stop after the frame in progress."""
        job = self._run_job
        if job is None or not job.pending:
            return
        job.cancel()
        self.cancel_button.setEnabled(False)
        self.stage_label.setText("Cancelling: the run stops after the frame "
                                 "or analysis in progress, and keeps what is "
                                 "finished")

    def is_running(self) -> bool:
        """True from Run until the run's result (or failure) arrives."""
        return self._run_job is not None and self._run_job.pending

    @Slot(int, int, str)
    def _on_run_progress(self, done: int, total: int, stage: str) -> None:
        if self._closing:
            return
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(min(done, max(1, total)))
        elapsed = self._run_job.elapsed_s() if self._run_job else 0.0
        prefix = "Cancelling; " if self._run_job and \
            self._run_job.cancel_requested else ""
        self.stage_label.setText(f"{prefix}{stage} ({elapsed:.0f} s)")

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
            self._update_actions()
            self.stage_label.setText(
                f"Run cancelled after {elapsed:.1f} s{extra}; the last "
                "complete run stays on show (Run > Show the cancelled run's "
                "partial result shows what was finished)")
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
        self.stage_label.setText(f"Run {state} in {elapsed:.1f} s"
                                 f"{extra}")
        self.finished.emit(result)

    def _show_stored(self, result) -> None:
        self.result = result
        self._show_result(result)
        self._frame_to_threshold_panel()
        self._update_actions()

    @Slot()
    def show_complete_result(self) -> bool:
        """Show the last run that finished uncancelled."""
        if self.complete_result is None:
            return False
        self._show_stored(self.complete_result)
        self.stage_label.setText("Showing the last complete run")
        return True

    @Slot()
    def show_partial_result(self) -> bool:
        """Show what the cancelled run finished."""
        if self.partial_result is None:
            return False
        self._show_stored(self.partial_result)
        self.stage_label.setText("Showing the cancelled run's partial "
                                 "result (the frames it finished)")
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
            # refused within a second rather than after every per-frame
            # analysis: the dynamics analyses say why, and Run runs the
            # others
            request = getattr(self._run_job, "request", None)
            reason = tracks[0].split(f": {md_jobs.TRACKS_INPUT}: ", 1)[1]
            if self.setup is not None and request is not None:
                self.setup.note_refusal(request, reason)
            text = ("The run did not start: the tracks the dynamics analyses "
                    f"read could not be collected ({reason}). Those analyses "
                    "now say so beside their boxes; Run analyses runs the "
                    "others")
        elif failure.kind == "request":
            text = "The run needs: " + "; ".join(failure.missing)
        else:
            text = f"The run stopped: {failure.message}"
        self.stage_label.setText(text)
        notes = text + ("\n\n" + failure.detail if failure.detail else "")
        if self.result is not None:
            # the run on show stays on show, with its notes and provenance
            notes += ("\n\nThe results on show are those of an earlier run, "
                      "whose notes follow.\n\n"
                      + "\n".join(self.notes_lines(self.result)))
        self.notes_text.setPlainText(notes)
        self.finished.emit(failure)

    @Slot()
    def _on_run_ended(self) -> None:
        if self._closing:
            return
        # the thread of a run has ended; a newer run may already be going
        self._set_running(self.is_running())

    def _set_running(self, running: bool) -> None:
        self.progress.setVisible(running)
        self.cancel_button.setVisible(running)
        self.cancel_button.setEnabled(running)
        if not running:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
        if self.setup is not None:
            self.setup.set_running(running)
        if self.results_view is not None:
            # shown only while no run is on show
            self._set_empty_text(running)
        self._update_actions()

    @Slot()
    def _update_actions(self) -> None:
        if not hasattr(self, "help_action"):
            return              # the menus are being built
        running = self.is_running()
        self.run_action.setEnabled(self.setup is not None and not running)
        self.cancel_action.setEnabled(running)
        has = self.result is not None
        self.export_csv_action.setEnabled(has)
        self.export_xlsx_action.setEnabled(has)
        browser = self.results_view
        for mine, name in ((self.export_figure_action, "export_figure_action"),
                           (self.export_rows_action, "export_rows_action")):
            # the browser's own action knows whether the item shown has a
            # figure (or rows); without one, whether the browser saves at all
            theirs = getattr(browser, name, None)
            mine.setEnabled(has and (theirs.isEnabled() if theirs is not None
                                     else False))
        self.save_request_action.setEnabled(self.setup is not None)
        self.show_complete_action.setEnabled(
            self.complete_result is not None
            and self.result is not self.complete_result)
        self.show_partial_action.setEnabled(
            self.partial_result is not None
            and self.result is not self.partial_result)

    def _browser_call(self, name: str) -> None:
        method = getattr(self.results_view, name, None)
        if method is not None:
            method()
        else:
            self.stage_label.setText("The result list of this build saves "
                                     "no figure or rows; File > Export writes "
                                     "every table")

    @Slot()
    def _save_figure_shown(self) -> None:
        self._browser_call("choose_export_figure")

    @Slot()
    def _save_rows_shown(self) -> None:
        self._browser_call("choose_export_rows")

    @Slot()
    def _run_from_menu(self) -> None:
        self.run_analysis()

    @Slot()
    def show_manual(self) -> None:
        """The manual, open at its MD section (Help, F1)."""
        from .help import MD_SECTION, ManualDialog

        existing = self._manual
        try:
            if existing is not None and existing.isVisible():
                existing.show_section(MD_SECTION)
                existing.raise_()
                existing.activateWindow()
                return
        except RuntimeError:         # closed and deleted
            pass
        dialog = ManualDialog(self, section=MD_SECTION, theme=self.theme)
        self._manual = dialog
        dialog.show()

    @Slot(str)
    def _on_browser_status(self, text: str) -> None:
        if not self._closing:
            self.stage_label.setText(text)

    # -- results -------------------------------------------------------------
    def _show_result(self, result) -> None:
        view = self.results_view
        try:
            view.set_result(result)
        except Exception as error:         # noqa: BLE001 - shown, not raised
            fallback = _ResultList()
            fallback.set_result(result)
            fallback.note.setText(f"The result browser could not show this "
                                  f"run ({error}); the tables are listed "
                                  "here.")
            index = self.tabs.indexOf(view)
            self.tabs.removeTab(index)
            self.tabs.insertTab(index, fallback, "Results")
            self.results_view = fallback
        self.notes_text.setPlainText("\n".join(self.notes_lines(result)))
        self.tabs.setCurrentWidget(self.results_view)
        self._update_actions()

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
                # the analysis's own notes first, then its provenance; a
                # line the engine repeats (the reader's notes reach an
                # analysis through several of its parts) is listed once
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
        if self.summary is not None:
            lines.append("== the model, as read ==")
            lines.extend(self.summary.describe_lines)
        return lines

    def _show_model_notes(self) -> None:
        self.notes_text.setPlainText("\n".join(self.notes_lines(None)))

    # -- the 3D view and the threshold panel -----------------------------------
    def _first_frame(self) -> None:
        if self.summary is None or self._closing:
            return
        self.show_frame(0)

    def show_frame(self, k: int, v_bond_vu: float | None = None) -> None:
        """Load frame ``k`` (and its bonds, unless only the atoms are drawn)
        on a worker thread, then draw it and hand it to the threshold
        panel."""
        if self.trajectory is None or self.frame_tab is None:
            return
        if self._frame_job is not None and self._frame_job.pending:
            self._pending_frame = (k, v_bond_vu)
            return
        from ..core import bv

        setup = self.setup
        overrides = setup.oxidation.overrides() if setup else {}
        if setup and setup.oxidation.problems():
            self.frame_tab.note.setText("The frame is drawn once every "
                                        "element has an oxidation state.")
            return
        params = None
        if setup is not None:
            try:
                params = setup.params_field.value()
            except ValueError:
                params = None
            if params is setup.params_field.UNSET:
                params = None
        v_bond = float(v_bond_vu if v_bond_vu is not None else
                       (setup.v_bond.value() if setup else
                        bv.V_BOND_DEFAULT))
        v_list = float(setup.v_list.value() if setup else bv.V_LIST_DEFAULT)
        with_bonds = self.frame_tab.with_bonds()
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
        self.frame_tab.note.setText(f"Loading frame {k} …")
        job.start()

    @Slot(object)
    def _on_frame_loaded(self, fv) -> None:
        if self._closing or self.frame_tab is None:
            return
        job = self.sender()
        if getattr(job, "generation", self._model_generation) != \
                self._model_generation:
            return                  # a frame of the model as read before
        self.frame_view = fv
        if fv.scene is not None:
            self.frame_tab.view.set_scene(fv.scene)
        self.frame_tab.note.setText(f"frame {fv.k}: {fv.note} "
                                    f"({fv.seconds:.2f} s)")
        self._frame_to_threshold_panel()

    def _frame_to_threshold_panel(self) -> None:
        """Hand the frame on show to the threshold panel: with its valence
        table when one was built (no second search), the threshold it was
        drawn at, and the threshold of the last run, so the panel can name
        both."""
        fv = self.frame_view
        set_frame = getattr(self.threshold_panel, "set_frame", None)
        if fv is None or set_frame is None:
            return
        run_v = None
        if self.result is not None:
            run_v = float(self.result.request.v_bond_vu)
        try:
            try:
                set_frame(fv.frame, fv.ox_atom, fv.params,
                          v_list_vu=fv.v_list_vu, v_bond_vu=fv.v_bond_vu,
                          analysis_v_bond_vu=run_v, label=f"frame {fv.k}",
                          table=fv.table)
            except TypeError:
                set_frame(fv.frame, fv.ox_atom, fv.params)
        except Exception as error:         # noqa: BLE001 - shown, not raised
            self.stage_label.setText(f"The threshold panel could not take "
                                     f"frame {fv.k}: {error}")

    @Slot(object)
    def _on_frame_failed(self, failure) -> None:
        if self._closing or self.frame_tab is None:
            return
        self.frame_tab.note.setText(f"The frame was not loaded: "
                                    f"{failure.message}")

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
        if fv is not None and fv.bonds is None and self.frame_tab.with_bonds():
            self.show_frame(fv.k, fv.v_bond_vu)

    @Slot(float)
    def _on_panel_threshold(self, v_bond_vu: float) -> None:
        """The threshold panel moved v_bond: the drawn scene restyles its
        contacts (no search; contacts below v_list stay undrawn)."""
        fv = self.frame_view
        if self.frame_tab is not None and fv is not None \
                and fv.table is not None:
            self.frame_tab.view.set_threshold(float(v_bond_vu))

    # -- export --------------------------------------------------------------
    def export_results(self, path) -> md_jobs.Job | None:
        """Write the last result: a path ending in .xlsx is one workbook,
        any other path a directory of CSV files (md_export.export). Runs on
        a worker thread; returns the job (None when there is no result)."""
        if self.result is None:
            self.stage_label.setText("Nothing to export: no run has finished")
            return None
        if self._export_job is not None and self._export_job.pending:
            self.stage_label.setText("An export is in progress")
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
        self.stage_label.setText(f"Writing {target} …")
        job.start()
        return job

    @Slot(object)
    def _on_export_done(self, written) -> None:
        if self._closing:
            return
        paths = list(written or [])
        where = paths[0].parent if len(paths) > 1 else (paths[0] if paths
                                                         else "")
        self.stage_label.setText(f"Wrote {len(paths)} file(s) to {where}")

    @Slot(object)
    def _on_export_failed(self, failure) -> None:
        if self._closing:
            return
        self.stage_label.setText(f"The export stopped: {failure.message}")

    @Slot()
    def _choose_export_csv(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Export results as CSV files: choose a folder")
        if folder:
            stem = Path(md_jobs.source_label(self.source)).stem
            self.export_results(Path(folder) / f"{stem}_facet_md")

    @Slot()
    def _choose_export_xlsx(self) -> None:
        stem = Path(md_jobs.source_label(self.source)).stem
        path, _ = QFileDialog.getSaveFileName(
            self, "Export results as an XLSX workbook",
            f"{stem}_facet_md.xlsx", "Excel workbook (*.xlsx)")
        if path:
            if not path.lower().endswith(".xlsx"):
                path += ".xlsx"
            self.export_results(path)

    # -- the request as a file ----------------------------------------------
    def save_request(self, path) -> Path:
        """Write the setup as a request file (TOML) that ``py -3.11 -m
        facet.md analyse FILE --request`` reads; the first line names the
        command. ValueError while there is no setup (no model read)."""
        from .md_dialogs import request_toml

        if self.setup is None:
            raise ValueError("no model is read yet")
        source = self.source[0] if isinstance(self.source, list) \
            else str(self.source)
        text = request_toml(self.setup.request_spec(), source=source)
        target = Path(path)
        target.write_text(text, encoding="utf-8")
        return target

    @Slot()
    def _choose_save_request(self) -> None:
        if self.setup is None:
            self.stage_label.setText("No model is read yet: no request to "
                                     "save")
            return
        stem = Path(md_jobs.source_label(self.source)).stem
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the request for the command line",
            f"{stem}_request.toml", "Request file (*.toml)")
        if not path:
            return
        if not path.lower().endswith(".toml"):
            path += ".toml"
        try:
            written = self.save_request(path)
        except (OSError, ValueError) as error:
            self.stage_label.setText(f"The request was not written: {error}")
        else:
            self.stage_label.setText(
                f"Wrote {written}: py -3.11 -m facet.md analyse <the model "
                f"file> --request {written.name} repeats the run")

    # -- theme ---------------------------------------------------------------
    def apply_theme(self, theme) -> None:
        """Adopt ``theme`` in this window's own views (no app-wide restyle)."""
        self.theme = theme
        if self.frame_tab is not None:
            self.frame_tab.view.set_theme(theme)
            if self.frame_view is not None:
                self.show_frame(self.frame_view.k, self.frame_view.v_bond_vu)
        for panel in (self.results_view, self.threshold_panel):
            for name in ("apply_theme", "set_theme"):
                method = getattr(panel, name, None)
                if method is not None:
                    method(theme)
                    break

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
                            self._export_job) if j is not None and not j.done]

    def _busy(self) -> bool:
        """True while any job of this window has not delivered its result."""
        return any(j is not None and j.pending for j in (
            self._open_job, self._run_job, self._frame_job,
            self._export_job))

    def closeEvent(self, event) -> None:
        if self._closing:
            # closed already (a second close() before the deferred delete)
            super().closeEvent(event)
            return
        self._closing = True
        jobs = self._jobs()
        for job in jobs:
            job.cancel()
        deadline = time.perf_counter() + CLOSE_WAIT_MS / 1000.0

        def left_ms() -> int:
            return int(max(0.0, deadline - time.perf_counter()) * 1000)

        for job in jobs:
            job.wait(left_ms())
        for panel in (self.results_view, self.threshold_panel):
            shutdown = getattr(panel, "shutdown", None)
            if shutdown is not None:
                # the browser's export thread and the threshold panel's
                # search thread cannot be cancelled: each is waited for
                # within the same CLOSE_WAIT_MS, then left to finish in the
                # background. md_views keeps their threads until they end,
                # and Qt drops a queued result whose receiver is gone.
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
        still = [j for j in jobs if j.is_running()]
        if not still and self.trajectory is not None:
            close = getattr(self.trajectory, "close", None)
            if close is not None:
                close()
        if self in _WINDOWS:
            _WINDOWS.remove(self)
        holder = getattr(self._opener, "_model_windows", None)
        if isinstance(holder, list) and self in holder:
            holder.remove(self)
        self.closed.emit(self)
        super().closeEvent(event)
        # A closed window is freed: its widgets, its trajectory and its
        # results (27 MB and 844 widgets per 3 000-atom model, which every
        # app-wide restyle of the crystal window would otherwise go
        # through). The jobs still running hold what they work on.
        self.deleteLater()

    # -- drag and drop -------------------------------------------------------
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls() and all(
                u.isLocalFile() for u in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        paths = [u.toLocalFile() for u in event.mimeData().urls()
                 if u.isLocalFile()]
        if paths:
            event.acceptProposedAction()
            self.open_dropped(paths)

    def open_dropped(self, paths) -> list:
        """Files dropped on this window, routed as the crystal window routes
        them (``preview.read_or_route``): an MD model opens here when this
        window holds none yet, else in a new Model window (files of one
        snapshot-per-file series as one model, ``preview.model_groups``);
        a crystal structure goes to the crystal window that opened this
        one; a file neither reads is named in the status bar. Returns the
        Model windows that read the MD files."""
        try:
            from . import preview
        except ImportError:          # a build without the crystal window
            self.open_paths(paths if len(paths) > 1 else paths[0])
            return [self]
        models, crystals, failed = [], [], []
        for path in paths:
            kind, value = preview.read_or_route(path)
            if kind == "md":
                models.append((str(path), value))
            elif kind == "structure":
                crystals.append(str(path))
            else:
                failed.append(f"{Path(path).name}: {value}")
        opened = []
        for group in preview.model_groups(models) if models else []:
            opened.append(self.open_paths(group))
        texts = []
        if crystals:
            load = getattr(self._opener, "load_many", None)
            if load is not None and _alive(self._opener):
                load(crystals)
                self._opener.raise_()
                texts.append(f"{len(crystals)} crystal file(s) opened in the "
                             "crystal window")
            else:
                texts.append(", ".join(Path(p).name for p in crystals)
                             + ": a crystal structure, which opens in "
                             "FACET's crystal window, not in a Model window")
        if failed:
            texts.append("not read: " + "; ".join(failed))
        if texts:
            self.stage_label.setText("; ".join(texts))
        return opened


def open_model_window(paths, parent=None, *, read_options=None
                      ) -> ModelWindow:
    """Open ``paths`` in a new Model window, show it, and keep it alive
    (module list, and ``parent._model_windows`` when the parent has one)."""
    window = ModelWindow(paths, parent, read_options=read_options)
    _WINDOWS.append(window)
    holder = getattr(parent, "_model_windows", None)
    if isinstance(holder, list):
        holder.append(window)
    window.show()
    QTimer.singleShot(0, window.raise_)
    return window


def main(argv: Sequence[str] | None = None) -> int:
    """``py -3.11 -m facet.ui.md_workspace FILE``: one Model window."""
    import sys

    from PySide6.QtGui import QSurfaceFormat

    from ..gl import caps as caps_mod

    argv = list(sys.argv if argv is None else argv)
    QSurfaceFormat.setDefaultFormat(caps_mod.request_format())
    app = QApplication.instance() or QApplication(argv)
    chrome.apply(app, theme_mod.Theme())
    if len(argv) < 2:
        print("usage: python -m facet.ui.md_workspace FILE [FILE ...]")
        return 2
    open_model_window(argv[1:] if len(argv) > 2 else argv[1])
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
