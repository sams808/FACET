"""Per-site, per-element and per-atom style overrides.

The theme says how an element looks everywhere. This says how one site, or one
single atom, looks differently -- which is what you need the moment a structure
has two crystallographically distinct bismuth sites and the figure has to tell
them apart.

Three levels, each taking precedence over the one before: element, then site,
then the single atom. Only the fields actually set are applied, so giving a site
a colour leaves its radius following the theme, and clearing an override
restores what it was covering rather than a default.

Every override is listed with what it does, so a figure can be accounted for
later. Nothing here is applied silently.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import overrides as overrides_mod
from . import chrome


class OverridesPanel(QWidget):
    """Edit and review the overrides for the active structure."""

    changed = Signal(str)                   # a sentence for the undo history
    focusAtom = Signal(int)                 # show me this atom

    def __init__(self, parent=None):
        super().__init__(parent)
        self.structure = None
        self.overrides: overrides_mod.StyleOverrides | None = None
        self._loading = False
        self._build_layout()
        chrome.fit_every_combo(self)
        self.refresh()

    # -- layout ------------------------------------------------------------
    def _build_layout(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        hint = QLabel(
            "An override applies on top of the theme: element, then site, then "
            "the single atom. Only what you set is changed — giving a site a "
            "colour leaves its size following the theme. Right-click an atom in "
            "the 3D view for the same actions on whatever is under the pointer.")
        hint.setWordWrap(True)
        tiny = QFont(hint.font())
        tiny.setPointSizeF(max(7.0, tiny.pointSizeF() - 1.0))
        hint.setFont(tiny)
        chrome.mark_hint(hint)
        outer.addWidget(hint)

        outer.addWidget(self._editor())
        outer.addWidget(self._list(), 1)

    def _editor(self) -> QWidget:
        group = QGroupBox("Set an override")
        box = QVBoxLayout(group)
        box.setSpacing(4)

        target = QHBoxLayout()
        target.addWidget(QLabel("apply to"))
        self.level = QComboBox()
        self.level.addItem("site", "site")
        self.level.addItem("element", "element")
        self.level.currentIndexChanged.connect(self._refill_targets)
        target.addWidget(self.level)
        self.target = QComboBox()
        self.target.setMinimumWidth(130)
        target.addWidget(self.target, 1)
        box.addLayout(target)

        row = QHBoxLayout()
        self.colour_button = QPushButton("Colour…")
        self.colour_button.clicked.connect(self._set_colour)
        row.addWidget(self.colour_button)

        row.addWidget(QLabel("size ×"))
        self.radius = QDoubleSpinBox()
        self.radius.setRange(0.05, 6.0)
        self.radius.setSingleStep(0.05)
        self.radius.setDecimals(2)
        self.radius.setValue(1.0)
        row.addWidget(self.radius)
        self.apply_radius = QPushButton("Apply size")
        self.apply_radius.clicked.connect(self._set_radius)
        row.addWidget(self.apply_radius)

        self.hide_button = QPushButton("Hide")
        self.hide_button.clicked.connect(lambda: self._set_visible(False))
        row.addWidget(self.hide_button)
        self.show_button = QPushButton("Show")
        self.show_button.clicked.connect(lambda: self._set_visible(True))
        row.addWidget(self.show_button)
        row.addStretch(1)
        box.addLayout(row)

        second = QHBoxLayout()
        second.addWidget(QLabel("label as"))
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("leave empty to keep the site label")
        self.label_edit.returnPressed.connect(self._set_label)
        second.addWidget(self.label_edit, 1)
        apply_label = QPushButton("Apply label")
        apply_label.clicked.connect(self._set_label)
        second.addWidget(apply_label)
        clear = QPushButton("Clear this target")
        clear.clicked.connect(self._clear_target)
        second.addWidget(clear)
        box.addLayout(second)
        return group

    def _list(self) -> QWidget:
        group = QGroupBox("Overrides in force")
        box = QVBoxLayout(group)
        box.setSpacing(4)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["level", "target", "effect", ""])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch)
        font = QFont(self.table.font())
        font.setPointSizeF(max(7.5, font.pointSizeF() - 0.5))
        self.table.setFont(font)
        self.table.cellDoubleClicked.connect(self._on_double_click)
        box.addWidget(self.table, 1)

        row = QHBoxLayout()
        self.count_label = QLabel("")
        small = QFont(self.count_label.font())
        small.setPointSizeF(max(7.0, small.pointSizeF() - 1.0))
        self.count_label.setFont(small)
        chrome.mark_hint(self.count_label)
        row.addWidget(self.count_label, 1)
        clear_all = QPushButton("Remove all")
        clear_all.clicked.connect(self._clear_all)
        row.addWidget(clear_all)
        box.addLayout(row)
        return group

    # -- content -----------------------------------------------------------
    def set_context(self, structure, overrides) -> None:
        self.structure = structure
        self.overrides = overrides
        self._refill_targets()
        self.refresh()

    def _refill_targets(self) -> None:
        self._loading = True
        try:
            self.target.clear()
            if self.structure is None:
                return
            if self.level.currentData() == "element":
                for symbol in self.structure.elements_present:
                    self.target.addItem(symbol, symbol)
            else:
                for site in self.structure.sites:
                    self.target.addItem(f"{site.label}  ({site.element})",
                                        site.label)
        finally:
            self._loading = False

    def refresh(self) -> None:
        rows = []
        if self.overrides is not None:
            for symbol, style in sorted(self.overrides.by_element.items()):
                rows.append(("element", symbol, style, ("element", symbol)))
            for label, style in sorted(self.overrides.by_site.items()):
                rows.append(("site", label, style, ("site", label)))
            for index, style in sorted(self.overrides.by_atom.items()):
                name = f"atom {index}"
                if (self.structure is not None
                        and 0 <= index < len(self.structure.atoms)):
                    name = f"atom {index} ({self.structure.atoms[index].label})"
                rows.append(("atom", name, style, ("atom", index)))

        self.table.setRowCount(len(rows))
        for r, (level, target, style, key) in enumerate(rows):
            for c, text in enumerate((level, target, style.describe())):
                item = QTableWidgetItem(text)
                if c == 2 and style.color is not None:
                    item.setForeground(QColor(int(style.color[0] * 255),
                                              int(style.color[1] * 255),
                                              int(style.color[2] * 255)))
                self.table.setItem(r, c, item)
            remove = QPushButton("×")
            remove.setMaximumWidth(26)
            remove.setToolTip("Remove this override")
            remove.clicked.connect(lambda _=False, k=key: self._remove(k))
            self.table.setCellWidget(r, 3, remove)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch)

        total = self.overrides.count() if self.overrides is not None else 0
        self.count_label.setText(
            "no overrides; every atom follows the theme" if not total
            else f"{total} override{'s' if total != 1 else ''} in force")

    # -- editing -----------------------------------------------------------
    def _current(self):
        if self.overrides is None or self.target.count() == 0:
            return None, None
        return self.level.currentData(), self.target.currentData()

    def _write(self, description: str, **fields) -> None:
        level, target = self._current()
        if level is None or target is None:
            return
        if level == "element":
            self.overrides.set_element(target, **fields)
        else:
            self.overrides.set_site(target, **fields)
        self.refresh()
        self.changed.emit(f"{description} for {level} {target}")

    def _set_colour(self) -> None:
        level, target = self._current()
        if level is None:
            return
        existing = (self.overrides.by_element.get(target) if level == "element"
                    else self.overrides.by_site.get(target))
        initial = QColor(255, 255, 255)
        if existing is not None and existing.color is not None:
            initial = QColor(int(existing.color[0] * 255),
                             int(existing.color[1] * 255),
                             int(existing.color[2] * 255))
        chosen = QColorDialog.getColor(initial, self, f"Colour for {target}")
        if not chosen.isValid():
            return
        self._write("set the colour",
                    color=(chosen.redF(), chosen.greenF(), chosen.blueF()))

    def _set_radius(self) -> None:
        self._write("set the size", radius_scale=float(self.radius.value()))

    def _set_visible(self, visible: bool) -> None:
        self._write("hid" if not visible else "showed", visible=visible)

    def _set_label(self) -> None:
        text = self.label_edit.text().strip()
        self._write("set the label", label=text or None)

    def _clear_target(self) -> None:
        level, target = self._current()
        if level is None or self.overrides is None:
            return
        if level == "element":
            self.overrides.clear_element(target)
        else:
            self.overrides.clear_site(target)
        self.refresh()
        self.changed.emit(f"cleared the override on {level} {target}")

    def _remove(self, key) -> None:
        if self.overrides is None:
            return
        level, target = key
        if level == "element":
            self.overrides.clear_element(target)
        elif level == "site":
            self.overrides.clear_site(target)
        else:
            self.overrides.clear_atom(target)
        self.refresh()
        self.changed.emit(f"removed the override on {level} {target}")

    def _clear_all(self) -> None:
        if self.overrides is None or self.overrides.is_empty:
            return
        self.overrides.clear()
        self.refresh()
        self.changed.emit("removed every override")

    def _on_double_click(self, row: int, column: int) -> None:
        """Double-clicking an atom override asks the view to show that atom."""
        item = self.table.item(row, 1)
        level = self.table.item(row, 0)
        if item is None or level is None or level.text() != "atom":
            return
        text = item.text().split()
        try:
            self.focusAtom.emit(int(text[1]))
        except (IndexError, ValueError):
            pass
