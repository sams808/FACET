"""The Setup page of the Model workspace: a preset, the compact inputs, and
the analyses by question, over the real :class:`~facet.ui.md_dialogs.
SetupPanel`.

The panel is the engine of this page: it owns the oxidation table, the
former picker, the frame range, the thresholds, every option field and the
request assembly (``SetupPanel.refresh``, ``run_request``,
``missing_inputs``, the model's own reasons). This page shows the few
inputs a run usually needs in a column 450-500 px wide, moves the panel's
own widgets into that layout where they are the same input (the oxidation
table, the frame range, v_bond, v_list, the parameter file, the time-axis
fields, the 'No former to name' box, the read options), mirrors the
panel's analysis boxes in a tree grouped by question, and keeps the panel
itself hidden for everything else (``page.panel``: the option groups a
reason names, ``focus_field``, ``request_spec``).

Nothing is ticked at first: the formers are a model input and a preset is
an explicit user action (:mod:`facet.core.md_presets`). Any manual edit
flips the preset combo to Custom.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import md_analysis as ma, md_presets
from . import chrome
from .md_dialogs import (OPTION_GROUPS, Reason, SetupPanel, _hint,
                         group_class, label_of, module_of)

__all__ = ["SetupPage", "Chip", "ANALYSES_BY_QUESTION", "thousands"]

ANALYSIS_ROLE = Qt.UserRole + 1

# The analyses by the question they answer, in the order the page lists
# them; every name of md_analysis.ANALYSES not placed here goes under
# "Other", and a 'channels' analysis (when a build offers one) under voids.
ANALYSES_BY_QUESTION: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("Structure", (
        ("glass", "Coordination, speciation, Qn"),
        ("rings", "Rings"),
        ("components", "Connectivity, dimensionality"),
        ("coordination-sequences", "Coordination sequences"),
        ("polyhedral-sharing", "Polyhedral sharing"),
        ("warren-cowley", "Warren-Cowley order"),
        ("bond-order", "Steinhardt bond order"),
        ("tetrahedral-order", "Tetrahedral order"),
        ("polyhedron-shape", "Polyhedron shape"),
    )),
    ("Voids and channels", (
        ("voronoi", "Voronoi cells"),
        ("empty-spheres", "Empty spheres"),
        ("free-volume", "Free volume"),
        ("void-regions", "Void regions, elongation"),
        ("channels", "Channels by charge (bond-valence landscape)"),
        ("modifier-density", "Modifier-rich anions, clusters"),
    )),
    ("Compare with experiment", (
        ("scattering", "S(Q), G(r), total scattering"),
        ("scattering-comparison", "Scattering vs a measured curve"),
        ("nmr", "NMR shifts (correlation)"),
        ("nmr-comparison", "NMR fractions vs measured"),
        ("exafs", "EXAFS cumulants"),
        ("feff", "FEFF inputs"),
    )),
    ("Dynamics", (
        ("msd", "MSD, diffusion"),
        ("bond-lifetimes", "Bond lifetimes"),
        ("self-correlations", "Self correlations"),
        ("distinct-van-hove", "Distinct van Hove"),
        ("vacf", "VACF, VDOS"),
        ("kinetic-temperature", "Kinetic temperature"),
        ("conductivity", "Conductivity, Haven ratio"),
    )),
)


def thousands(n: int) -> str:
    """3 000: a thin space every three digits."""
    return f"{int(n):,}".replace(",", " ")


def _question_groups() -> list[tuple[str, list[tuple[str, str]]]]:
    """The groups with only the analyses this build offers, and an "Other"
    group for any analysis the table does not place."""
    placed = set()
    out = []
    for group, members in ANALYSES_BY_QUESTION:
        kept = [(n, t) for n, t in members if n in ma.ANALYSES]
        placed.update(n for n, _ in kept)
        if kept:
            out.append((group, kept))
    rest = [(n, n) for n in ma.ANALYSES if n not in placed]
    if rest:
        out.append(("Other", rest))
    return out


def _take(widget: QWidget) -> QWidget:
    """Remove ``widget`` from the layout and parent it sits in, so that
    another layout can take it (as ``SetupPanel.detach_footer`` does)."""
    parent = widget.parentWidget()
    if parent is not None and parent.layout() is not None:
        parent.layout().removeWidget(widget)
    widget.setParent(None)
    return widget


def _slot() -> tuple[QWidget, QHBoxLayout]:
    """A zero-margin holder a moved widget is put in."""
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(0)
    return holder, row


class Chip(QToolButton):
    """A former: a rounded, checkable button with the element symbol."""

    def __init__(self, text: str, tip: str = "", parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self.setObjectName("chip")
        self.setToolTip(tip)
        self.setCursor(Qt.PointingHandCursor)


def _chip_stylesheet(theme) -> str:
    ui = chrome.ui_colors(theme)
    return (f"QToolButton#chip {{ border: 1px solid {ui.border}; "
            f"border-radius: 10px; padding: 2px 10px; background: {ui.base}; "
            f"color: {ui.text}; }}"
            f"QToolButton#chip:checked {{ background: {ui.accent}; "
            f"color: {ui.accent_text}; border-color: {ui.accent}; }}"
            f"QToolButton#chip:disabled {{ color: {ui.muted}; }}")


# What a runnable analysis runs with, read from the panel's fields: the
# fields shown (group, name) and how the values are joined. A field left
# blank shows its default (the engine's), so the note never hides a value.
_INPUT_FIELDS = {
    "rings": (("network", "ring_criterion"), ("network", "ring_max_size")),
    "coordination-sequences": (("network", "n_shells"),),
    "bond-order": (("order", "degrees"),),
    "tetrahedral-order": (("order", "q_tet_selection"),),
    "polyhedron-shape": (("order", "neighbours"),),
    "voronoi": (("order", "min_face_area_ang2"),),
    "empty-spheres": (("voids", "radii"),),
    "free-volume": (("voids", "radii"), ("voids", "probe_radius_ang"),
                    ("voids", "grid_spacing_ang")),
    "scattering": (("scattering", "radiations"), ("scattering", "r_window")),
    "scattering-comparison": (("scattering", "r_window"),),
    "nmr": (("nmr", "lineshape"), ("nmr", "fwhm_ppm")),
    "exafs": (("exafs", "absorber"),),
    "feff": (("exafs", "absorber"), ("feff", "edge")),
    "msd": (("dynamics", "unwrap"),),
    "self-correlations": (("dynamics", "lag_t_ps"),),
    "distinct-van-hove": (("dynamics", "lag_t_ps"),),
    "vacf": (("dynamics", "velocities"),),
    "kinetic-temperature": (("dynamics", "velocities"),),
    "conductivity": ((None, "charges_e"), (None, "temperature_k")),
    "bond-lifetimes": (("dynamics", "gap_tolerance_frames"),),
}
_STATIC_NOTE = {"glass": "bond valence and distance"}
_SHORT_LABEL = {"ring_criterion": "", "ring_max_size": "≤ {} nodes",
                "n_shells": "{} shells", "degrees": "l = {}",
                "q_tet_selection": "", "neighbours": "{} neighbours",
                "min_face_area_ang2": "faces ≥ {} Å²", "radii": "{} radii",
                "probe_radius_ang": "probe {} Å", "grid_spacing_ang": "grid {} Å",
                "radiations": "", "r_window": "{} window",
                "lineshape": "", "fwhm_ppm": "{} ppm", "absorber": "absorber {}",
                "edge": "{} edge", "unwrap": "unwrap {}", "lag_t_ps": "lags {} ps",
                "velocities": "velocities: {}", "charges_e": "charges {}",
                "temperature_k": "{} K", "gap_tolerance_frames": "gap {} frames"}


class SetupPage(QWidget):
    """The Setup page: preset, composition, formers, frames, threshold, a
    folded "More…", the analyses tree, and a footer the window pins under
    the scroll area (:meth:`detach_footer`).

    Signals: ``changed()`` after every refresh of the panel;
    ``runRequested()`` from the Run button; ``cancelRequested()`` from the
    Cancel button; ``rereadRequested(dict)`` with the read options to read
    the file again with; ``fieldRequested(str)`` when the user asks for a
    field this page does not show ('group.name', the panel's
    ``focus_field`` takes it).
    """

    changed = Signal()
    runRequested = Signal()
    cancelRequested = Signal()
    rereadRequested = Signal(object)
    fieldRequested = Signal(str)

    def __init__(self, summary, trajectory=None, *, theme=None, parent=None):
        super().__init__(parent)
        self.summary = summary
        self.trajectory = trajectory
        self.theme = theme
        self._running = False
        self._applying = False
        self._syncing = False
        self._timestep_inline = False
        self.preset_notes: tuple[str, ...] = ()
        # the panel is a hidden child: it dies with the page, and its timer
        # with it, so no refresh runs on widgets the page took and freed
        self.panel = SetupPanel(summary, trajectory, theme=theme,
                                type_map=summary.type_map, parent=self)
        self.panel.hide()
        self.panel.changed.connect(self._on_panel_changed)
        self.panel.rereadRequested.connect(self.rereadRequested)
        self._muted = QBrush(QColor(chrome.ui_colors(theme).muted))
        self.setStyleSheet(_chip_stylesheet(theme))

        column = QVBoxLayout(self)
        column.setContentsMargins(8, 8, 8, 4)
        column.setSpacing(6)
        self.grid = grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(1, 1)

        # preset
        self.preset = QComboBox()
        for preset in md_presets.PRESETS:
            self.preset.addItem(preset.name)
            self.preset.setItemData(self.preset.count() - 1,
                                    preset.description, Qt.ToolTipRole)
        self.preset.setCurrentText("Custom")
        self.preset.setToolTip(
            "A preset ticks the analyses a question needs, names the "
            "formers among the cations present and fills the method inputs "
            "they require (ring criterion, void radii, …); nothing physical "
            "is filled in. Custom leaves everything as it is. Every value "
            "stays editable below.")
        chrome.fit_combo(self.preset)
        preset_row = QHBoxLayout()
        preset_row.setSpacing(8)
        preset_row.addWidget(self.preset)
        self.preset_note = _hint("ticks analyses and formers")
        preset_row.addWidget(self.preset_note, 1)
        grid.addWidget(self._label("Preset"), 0, 0)
        grid.addLayout(preset_row, 0, 1)

        # composition, from the file
        comp = " · ".join(f"{s} {thousands(n)}"
                          for s, n in summary.composition.items())
        self.composition = QLabel(f"{comp} · {summary.file_format}")
        self.composition.setWordWrap(True)
        self.composition.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.composition.setToolTip("\n".join(summary.describe_lines))
        grid.addWidget(self._label("Composition", "Frame 0, as read; the "
                                   "whole load summary is under More…"),
                       1, 0)
        grid.addWidget(self.composition, 1, 1)

        # formers as chips, mirroring the panel's picker
        self.chips: dict[str, Chip] = {}
        self._chip_cations: tuple = ()
        self.chip_row = QHBoxLayout()
        self.chip_row.setSpacing(6)
        self.former_note = _hint("tick the formers")
        self.former_note.setToolTip(self.panel.formers.note.text())
        grid.addWidget(self._label("Formers", "The network formers; the "
                                   "former-dependent descriptors count over "
                                   "them. None ticked greys the analyses "
                                   "that need them."), 2, 0)
        grid.addLayout(self.chip_row, 2, 1)
        self.panel.formers.changed.connect(self._sync_chips)

        # frames: the panel's FrameRange, with a one-line note
        self.frame_range = _take(self.panel.frame_range)
        self.frame_range.note.hide()
        self.frames_note = _hint()
        self.frames_note.setToolTip("The frames the run reads; the "
                                    "timesteps are the file's.")
        frames_col = QVBoxLayout()
        frames_col.setContentsMargins(0, 0, 0, 0)
        frames_col.setSpacing(2)
        frames_col.addWidget(self.frame_range)
        frames_col.addWidget(self.frames_note)
        grid.addWidget(self._label("Frames", "first, last and every n-th "
                                   "readable frame of the file"), 3, 0)
        grid.addLayout(frames_col, 3, 1)

        # threshold: the panel's v_bond
        self.v_bond = _take(self.panel.v_bond)
        self.v_bond.setToolTip("v_bond_vu: a contact above it is a bond "
                               "(CN, Qn, BO). The strip under the view "
                               "shows what each value counts.")
        self.threshold_note = _hint()
        thr_row = QHBoxLayout()
        thr_row.setSpacing(8)
        thr_row.addWidget(self.v_bond)
        thr_row.addWidget(self.threshold_note, 1)
        grid.addWidget(self._label("Threshold", "v_bond: the valence above "
                                   "which a contact is a bond"), 4, 0)
        grid.addLayout(thr_row, 4, 1)

        # the time axis, shown here only when a preset needs one the file
        # lacks (the field lives under More… otherwise)
        self.time_label = self._label("Time axis", "The MD timestep in fs: "
                                      "the file's timesteps times it give "
                                      "each frame's time.")
        self.time_holder, time_row = _slot()
        time_col = QVBoxLayout()
        time_col.setContentsMargins(0, 0, 0, 0)
        time_col.setSpacing(2)
        self.time_note = _hint()
        time_col.addWidget(self.time_holder)
        time_col.addWidget(self.time_note)
        self.time_widget = QWidget()
        self.time_widget.setLayout(time_col)
        grid.addWidget(self.time_label, 5, 0)
        grid.addWidget(self.time_widget, 5, 1)
        self.time_label.hide()
        self.time_widget.hide()
        column.addLayout(grid)

        # everything else, folded
        self.more = chrome.disclosure(
            "More: oxidation states, v_list, parameters, time axis, read "
            "options…", self._build_more())
        column.addWidget(self.more)

        # the analyses, by question
        head = QHBoxLayout()
        head.addWidget(QLabel("<b>Analyses</b>"))
        hint = _hint("by question; tick to run, a group ticks its children")
        hint.setToolTip("The second column is what a runnable analysis runs "
                        "with, or what a greyed one needs; double-click an "
                        "analysis to go to its inputs.")
        head.addWidget(hint, 1)
        column.addLayout(head)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setIndentation(18)
        self.tree.setTextElideMode(Qt.ElideRight)
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setMaximumSectionSize(210)
        self.tree.setMinimumHeight(180)
        self.tree.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.groups: dict[str, QTreeWidgetItem] = {}
        self.items: dict[str, QTreeWidgetItem] = {}
        self.question_of: dict[str, str] = {}
        bold = QFont(self.font())
        bold.setBold(True)
        for group, members in _question_groups():
            top = QTreeWidgetItem([group, ""])
            top.setFont(0, bold)
            top.setFlags(top.flags() | Qt.ItemIsUserCheckable
                         | Qt.ItemIsAutoTristate)
            top.setCheckState(0, Qt.Unchecked)
            top.setForeground(1, self._muted)
            self.tree.addTopLevelItem(top)
            self.groups[group] = top
            for name, text in members:
                item = QTreeWidgetItem([text, ""])
                item.setData(0, ANALYSIS_ROLE, name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(0, Qt.Unchecked)
                item.setForeground(1, self._muted)
                top.addChild(item)
                self.items[name] = item
                self.question_of[name] = group
            top.setExpanded(True)
        self.tree.itemChanged.connect(self._on_item)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        column.addWidget(self.tree, 1)

        # the footer, pinned by the window under the scroll area
        self.footer = QWidget()
        foot = QVBoxLayout(self.footer)
        foot.setContentsMargins(8, 4, 8, 6)
        foot.setSpacing(3)
        run_row = QHBoxLayout()
        # 'y': a letter no menu-bar title takes (File, Edit, View, Model, Help)
        self.run_button = QPushButton("Run anal&yses")
        self.run_button.setToolTip("Run the ticked analyses (Ctrl+R).")
        self.run_button.clicked.connect(self._on_run)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(True)
        self.progress.setMinimumWidth(150)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setToolTip("Stop after the current frame; what "
                                      "was finished is kept, marked as a "
                                      "cancelled run.")
        self.cancel_button.clicked.connect(self.cancelRequested)
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.progress, 3)
        run_row.addWidget(self.cancel_button)
        run_row.addStretch(1)
        foot.addLayout(run_row)
        self.run_note = _hint()
        foot.addWidget(self.run_note)
        column.addWidget(self.footer)

        # user edits flip the preset to Custom
        self.frame_range.changed.connect(self._user_edit)
        self.v_bond.valueChanged.connect(self._user_edit)
        self.v_list.valueChanged.connect(self._user_edit)
        self.panel.oxidation.changed.connect(self._user_edit)
        self.params_field.changed.connect(self._user_edit)
        self.timestep_field.changed.connect(self._user_edit)
        self.interval_field.changed.connect(self._user_edit)
        self.none_box.toggled.connect(self._user_edit)
        self.preset.currentTextChanged.connect(self._on_preset)

        self._sync_chips()
        self.set_running(False)
        self._sync_from_panel()

    # -- building --------------------------------------------------------------
    @staticmethod
    def _label(text: str, tip: str = "") -> QLabel:
        label = QLabel(text)
        label.setToolTip(tip)
        label.setAlignment(Qt.AlignRight | Qt.AlignTop)
        label.setContentsMargins(0, 4, 0, 0)
        return label

    def _build_more(self) -> QWidget:
        panel = self.panel
        more = QWidget()
        more_col = QVBoxLayout(more)
        more_col.setContentsMargins(0, 0, 0, 0)
        more_col.setSpacing(6)
        self.oxidation = _take(panel.oxidation)
        self.ox_notes = _take(panel.ox_notes)
        more_col.addWidget(QLabel("Oxidation states (model inputs, never "
                                  "resolved from the geometry)"))
        more_col.addWidget(self.oxidation)
        more_col.addWidget(self.ox_notes)
        more_grid = QGridLayout()
        more_grid.setHorizontalSpacing(10)
        more_grid.setVerticalSpacing(6)
        more_grid.setColumnStretch(1, 1)
        self.v_list = _take(panel.v_list)
        self.params_field = _take(panel.params_field)
        dynamics = panel.groups["md_dynamics"]
        self.timestep_field = _take(dynamics.fields[(None, "timestep_fs")])
        self.interval_field = _take(dynamics.fields[(None,
                                                     "frame_interval_ps")])
        self.timestep_field.widget.setPlaceholderText(
            "fs per MD step (blank: the file's frame times, if any)")
        self.interval_field.widget.setPlaceholderText(
            "ps between the frames written (blank: not given)")
        self.more_time_holder, more_time_row = _slot()
        more_time_row.addWidget(self.timestep_field)
        rows = ((label_of("v_list_vu"), self.v_list, "v_list_vu"),
                ("parameter file", self.params_field, "params"),
                (label_of("timestep_fs"), self.more_time_holder,
                 "timestep_fs"),
                (label_of("frame_interval_ps"), self.interval_field,
                 "frame_interval_ps"))
        for row, (text, widget, where) in enumerate(rows):
            label = self._label(text, f"Request name: {where}")
            more_grid.addWidget(label, row, 0)
            more_grid.addWidget(widget, row, 1)
        more_col.addLayout(more_grid)
        self.none_box = _take(panel.formers.none_box)
        more_col.addWidget(self.none_box)
        self.read_box = panel.read_box
        if self.read_box is not None:
            more_col.addWidget(_take(self.read_box))
        self.summary_text = _take(panel.summary_text)
        self.summary_fold = chrome.disclosure(
            "Load summary (what the reader read and assumed)",
            self.summary_text)
        more_col.addWidget(self.summary_fold)
        return more

    def detach_footer(self) -> QWidget:
        """Take the Run button, the progress bar, Cancel and the run line
        out of this page, for the window to put under the scroll area."""
        self.layout().removeWidget(self.footer)
        self.footer.setParent(None)
        return self.footer

    # -- state ---------------------------------------------------------------
    def ticked(self) -> list[str]:
        """The analyses ticked and enabled, in the engine's order."""
        return [n for n in ma.ANALYSES
                if n in self.items
                and self.items[n].checkState(0) == Qt.Checked
                and bool(self.items[n].flags() & Qt.ItemIsEnabled)]

    def formers(self) -> frozenset[str]:
        return frozenset(self.panel.formers.ticked())

    def refresh(self) -> None:
        """Rebuild the request now (the panel waits 120 ms after an edit)."""
        self.panel.refresh()

    def request(self):
        """The AnalysisRequest to run; ``md_analysis.RequestError`` with the
        line the footer shows when the run cannot start."""
        request = self.panel.request()
        if request is None:
            ok, text = self._status()
            missing = [ma.MissingInput("setup", "run", p)
                       for p in dict.fromkeys(self.panel.problems())] or [
                ma.MissingInput("setup", "run", text)]
            error = ma.RequestError(missing)
            error.args = (text,)
            raise error
        return request

    def can_run(self) -> tuple[bool, str]:
        """(whether Run is offered, the line the footer shows)."""
        self.panel.problems()           # flushes a pending refresh
        return self._status()

    def _status(self) -> tuple[bool, str]:
        if self._running:
            return False, ("A run is in progress; the results are listed in "
                           "Results when it ends.")
        request = self.panel._request
        if request is None:
            problems = list(dict.fromkeys(self.panel._problems))
            return False, "Run needs: " + ("; ".join(problems) or
                                           "an analysis ticked") + "."
        formers = self.panel.formers.formers()
        if formers == ma.NO_FORMERS:
            former_text = "none named"
        elif formers:
            former_text = ", ".join(sorted(formers))
        else:
            former_text = "none"
        return True, (f"Runs {', '.join(request.analyses)} on "
                      f"{self.frame_range.count()} frame(s); formers "
                      f"{former_text}.")

    def set_running(self, on: bool) -> None:
        self._running = bool(on)
        self.panel.set_running(on)
        self.progress.setVisible(on)
        self.cancel_button.setVisible(on)
        if not on:
            self.progress.setValue(0)
        self._describe()

    def set_progress(self, done: int, total: int, stage: str = "") -> None:
        total = max(int(total), 1)
        self.progress.setRange(0, total)
        self.progress.setValue(min(int(done), total))
        percent = 100.0 * min(int(done), total) / total
        self.progress.setFormat(f"{percent:.0f} %  ·  {stage}" if stage
                                else f"{percent:.0f} %")

    def apply_theme(self, theme) -> None:
        self.theme = theme
        self.panel.theme = theme
        for group in self.panel.groups.values():
            group.theme = theme
        self.setStyleSheet(_chip_stylesheet(theme))
        self._muted = QBrush(QColor(chrome.ui_colors(theme).muted))
        for item in list(self.groups.values()) + list(self.items.values()):
            item.setForeground(1, self._muted)
        self.panel.refresh()

    # -- presets -------------------------------------------------------------
    @Slot(str)
    def _on_preset(self, name: str) -> None:
        self.apply_preset(name)

    def apply_preset(self, name: str):
        """Apply the preset ``name`` (:func:`md_presets.apply_preset`) and
        return the PresetApplication; the combo follows."""
        preset = md_presets.preset_named(name)
        applied = md_presets.apply_preset(preset, self.summary,
                                          self.panel.oxidation.states())
        self.preset_notes = applied.notes
        if self.preset.currentText() != name:
            self.preset.blockSignals(True)
            self.preset.setCurrentText(name)
            self.preset.blockSignals(False)
        if not applied.changes:
            self.preset_note.setText("the ticks and fields as edited")
            self.preset_note.setToolTip(preset.description)
            return applied
        self._applying = True
        try:
            for key, text in applied.options.items():
                group, _, field_name = key.partition(".")
                try:
                    self.panel.field(group or None, field_name).set_text(text)
                except (KeyError, ValueError):
                    pass
            if applied.formers:
                self.panel.formers.set_formers(applied.formers)
            self.panel.set_analyses(applied.analyses)
            inline = ("time axis" in preset.needs
                      and not self.summary.has_times
                      and self.summary.n_frames > 1
                      and self.timestep_field.is_blank()
                      and self.interval_field.is_blank())
            self._place_timestep(inline)
        finally:
            self._applying = False
        n_notes = len(applied.notes)
        self.preset_note.setText(
            f"ticks {len(applied.analyses)} analyses"
            + (f"; formers {', '.join(sorted(applied.formers))}"
               if applied.formers else "")
            + (f" · {n_notes} note(s)" if n_notes else ""))
        self.preset_note.setToolTip("\n".join(applied.notes)
                                    or preset.description)
        everything = preset.name == "Everything"
        ticked = set(applied.analyses)
        for group, top in self.groups.items():
            members = [n for n, g in self.question_of.items() if g == group]
            top.setExpanded(everything or any(n in ticked for n in members))
        return applied

    def _flip_to_custom(self) -> None:
        if self._applying or self.preset.currentText() == "Custom":
            return
        self.preset.blockSignals(True)
        self.preset.setCurrentText("Custom")
        self.preset.blockSignals(False)
        self.preset_notes = ()
        self.preset_note.setText("the ticks and fields as edited")
        self.preset_note.setToolTip(md_presets.preset_named("Custom")
                                    .description)

    @Slot()
    def _user_edit(self, *_args) -> None:
        self._flip_to_custom()

    # -- the time axis field ---------------------------------------------------
    def _place_timestep(self, inline: bool) -> None:
        """Put the timestep field in the header (``inline``) or under More…."""
        if inline == self._timestep_inline:
            if inline:
                self._time_reason()
            return
        _take(self.timestep_field)
        target = self.time_holder if inline else self.more_time_holder
        target.layout().addWidget(self.timestep_field)
        self._timestep_inline = inline
        self.time_label.setVisible(inline)
        self.time_widget.setVisible(inline)
        if inline:
            self._time_reason()

    def _time_reason(self) -> None:
        text = ("the file states no frame times: give the MD timestep in fs "
                "(or the time between frames, under More…)")
        for reasons in self.panel._reasons.values():
            for reason in reasons:
                if isinstance(reason, Reason) and \
                        reason.where == "timestep_fs":
                    # the engine's first sentence; the rest is in the tip
                    text = "needs " + reason.text.split(". ")[0].rstrip(".") \
                        + "."
                    self.time_note.setToolTip(reason.text)
                    break
        self.time_note.setText(text)

    # -- chips -----------------------------------------------------------------
    @Slot()
    def _sync_chips(self) -> None:
        picker = self.panel.formers
        cations = tuple(picker.cations())
        if cations != self._chip_cations:
            for chip in self.chips.values():
                self.chip_row.removeWidget(chip)
                chip.deleteLater()
            self.chips = {}
            self.chip_row.removeWidget(self.former_note)
            for symbol in cations:
                chip = Chip(symbol, f"{symbol}: tick to count it as a "
                                    "network former (Qn, rings, connectivity "
                                    "are counted over the formers).")
                chip.toggled.connect(self._on_chip)
                self.chips[symbol] = chip
                self.chip_row.addWidget(chip)
            self.chip_row.addWidget(self.former_note, 1)
            self._chip_cations = cations
        ticked = picker.ticked()
        for symbol, chip in self.chips.items():
            chip.blockSignals(True)
            chip.setChecked(symbol in ticked)
            chip.blockSignals(False)
        value = picker.formers()
        if value == ma.NO_FORMERS:
            self.former_note.setText("no former to name (More…)")
        elif value:
            self.former_note.setText("formers " + ", ".join(sorted(value)))
        else:
            self.former_note.setText("tick the formers" if cations else
                                     "no cation with a state (More…)")
        self.former_note.setToolTip(picker.note.text())

    def _on_chip(self, _on: bool) -> None:
        wanted = {s for s, c in self.chips.items() if c.isChecked()}
        self.panel.formers.set_formers(wanted)
        self._flip_to_custom()

    # -- the tree --------------------------------------------------------------
    def _on_item(self, item: QTreeWidgetItem, _column: int) -> None:
        if self._syncing:
            return
        name = item.data(0, ANALYSIS_ROLE)
        self._syncing = True
        self.tree.blockSignals(True)
        try:
            # a parent tick reaches disabled children: untick them again
            for child in self.items.values():
                if not (child.flags() & Qt.ItemIsEnabled) and \
                        child.checkState(0) != Qt.Unchecked:
                    child.setCheckState(0, Qt.Unchecked)
            if name is None:
                for child_name in [n for n, g in self.question_of.items()
                                   if self.groups[g] is item]:
                    self._push_box(child_name)
            else:
                self._push_box(name)
            self._sync_groups()
        finally:
            self.tree.blockSignals(False)
            self._syncing = False
        self._flip_to_custom()
        self._describe()

    def _push_box(self, name: str) -> None:
        item = self.items[name]
        if not (item.flags() & Qt.ItemIsEnabled):
            return
        box = self.panel.box(name)
        on = item.checkState(0) == Qt.Checked
        if box.isChecked() != on:
            box.setChecked(on)

    def _sync_groups(self) -> None:
        for group, top in self.groups.items():
            members = [n for n, g in self.question_of.items() if g == group]
            on = sum(self.items[n].checkState(0) == Qt.Checked
                     for n in members)
            top.setText(1, f"{on} of {len(members)}")
            top.setCheckState(0, Qt.Checked if on == len(members) else
                              Qt.PartiallyChecked if on else Qt.Unchecked)

    @Slot()
    def _on_panel_changed(self) -> None:
        self._sync_from_panel()

    def _sync_from_panel(self) -> None:
        """Mirror the panel's boxes and reasons in the tree."""
        self._syncing = True
        self.tree.blockSignals(True)
        try:
            reasons = self.panel._reasons
            for name, item in self.items.items():
                box = self.panel.box(name)
                enabled = box.isEnabled()
                flags = item.flags()
                item.setFlags(flags | Qt.ItemIsEnabled if enabled
                              else flags & ~Qt.ItemIsEnabled)
                item.setCheckState(0, Qt.Checked if enabled and
                                   box.isChecked() else Qt.Unchecked)
                summary = ma.ANALYSIS_SUMMARIES.get(name, "")
                lacking = reasons.get(name) or []
                if lacking:
                    item.setText(1, self._short_reason(lacking))
                    tip = (f"{name}: {summary}\n\nGreyed: needs "
                           + "; ".join(str(r) for r in lacking) + ".")
                else:
                    item.setText(1, self._input_note(name))
                    long = self._input_note(name, long=True)
                    tip = (f"{name}: {summary}\n\n"
                           + (f"Runs with {long}. " if long else "")
                           + "Double-click to go to its inputs.")
                item.setToolTip(0, tip)
                item.setToolTip(1, tip)
            self._sync_groups()
        finally:
            self.tree.blockSignals(False)
            self._syncing = False
        if self._timestep_inline:
            self._time_reason()
        self._describe()
        self.changed.emit()

    @staticmethod
    def _short_reason(lacking: Sequence) -> str:
        parts = []
        for reason in lacking:
            link = getattr(reason, "link", "")
            text = str(reason)
            if link:
                parts.append(link)
            else:
                words = text.split()
                parts.append(" ".join(words[:3]) + ("…" if len(words) > 3
                                                     else ""))
        seen = list(dict.fromkeys(parts))
        return "needs " + ", ".join(seen[:2]) + (" +" + str(len(seen) - 2)
                                                 if len(seen) > 2 else "")

    def _field_text(self, group, name) -> str:
        try:
            field_ = self.panel.field(group, name)
        except KeyError:
            return ""
        text = field_.text()
        if text == "" and field_.default is not None and field_.default != ():
            default = field_.default
            text = ", ".join(str(v) for v in default) \
                if isinstance(default, (tuple, list)) else f"{default}"
        return text

    def _input_note(self, name: str, *, long: bool = False) -> str:
        parts = []
        if name in _STATIC_NOTE:
            parts.append(_STATIC_NOTE[name])
        for group, field_name in _INPUT_FIELDS.get(name, ()):
            text = self._field_text(group, field_name)
            if not text:
                continue
            if long:
                parts.append(f"{label_of(field_name)} {text}")
            else:
                pattern = _SHORT_LABEL.get(field_name, "{}")
                parts.append(pattern.format(text) if pattern else text)
        reads = self.panel.reason_text(name)
        if reads.startswith("runs with "):
            what = reads[len("runs with "):].split(" (")[0]
            parts.append(f"reads {what}")
        return ", ".join(parts)

    def _on_double_click(self, item: QTreeWidgetItem, _column: int) -> None:
        name = item.data(0, ANALYSIS_ROLE)
        if name is None:
            return
        where = None
        for reason in self.panel._reasons.get(name) or []:
            if getattr(reason, "where", None):
                where = reason.where
                break
        if where is None:
            fields = _INPUT_FIELDS.get(name)
            if fields:
                group, field_name = fields[0]
                where = f"{group}.{field_name}" if group else field_name
            else:
                for group in OPTION_GROUPS.get(module_of(name), ()):
                    first = dataclasses.fields(group_class(group))[0].name
                    where = f"{group}.{first}"
                    break
        if where is None:
            return
        self.reveal(where)

    def reveal(self, where: str) -> bool:
        """Show the field a reason names when this page holds it (frames,
        the thresholds, the time axis, the parameter file: under More… or
        inline); ``fieldRequested(where)`` is emitted in every case, for
        the window to show the panel's field when it is not here."""
        handled = True
        if where == "frames":
            self.frame_range.first.setFocus(Qt.OtherFocusReason)
        elif where == "formers":
            pass
        elif where == "v_bond_vu":
            self.v_bond.setFocus(Qt.OtherFocusReason)
        elif where == "timestep_fs" and self._timestep_inline:
            self.timestep_field.widget.setFocus(Qt.OtherFocusReason)
        elif where in ("timestep_fs", "frame_interval_ps", "params",
                       "v_list_vu"):
            self.more.button.setChecked(True)
            target = {"timestep_fs": self.timestep_field,
                      "frame_interval_ps": self.interval_field,
                      "params": self.params_field,
                      "v_list_vu": self.v_list}[where]
            widget = getattr(target, "widget", target)
            widget.setFocus(Qt.OtherFocusReason)
        else:
            handled = False
        self.fieldRequested.emit(where)
        return handled

    # -- the footer ------------------------------------------------------------
    def _describe(self) -> None:
        summary = self.summary
        count = self.frame_range.count()
        chosen = list(self.frame_range.indices())
        from ..core.md_model import NO_TIMESTEP

        steps = summary.timesteps[chosen] if chosen else []
        known = [int(s) for s in steps if int(s) != NO_TIMESTEP]
        if known:
            span = f"timesteps {thousands(min(known))}–{thousands(max(known))}"
        elif chosen:
            span = "no timesteps in the file"
        else:
            span = "no frame"
        times = ("frame times in the file" if summary.has_times
                 else "no frame times in the file")
        self.frames_note.setText(f"{count} of {summary.n_frames} frames · "
                                 f"{span} · {times}")
        params = self.params_field.text()
        self.threshold_note.setText(
            f"v_list {self.v_list.value():g} v.u. · "
            + (f"parameters {params}" if params else "default parameters"))
        ok, text = self._status()
        self.run_button.setEnabled(ok)
        self.run_note.setText(text)
        self.run_note.setToolTip(text)

    @Slot()
    def _on_run(self) -> None:
        ok, _ = self.can_run()
        if ok:
            self.runRequested.emit()
