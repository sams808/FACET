"""The window's frame: the pieces of layout every workspace shares.

Three widgets, none of which knows anything about crystallography:

* :class:`ReflowBar` -- a toolbar made of groups of controls, a thin rule
  between them, that sits on one row while the window is wide enough and
  moves its trailing groups to a second and a third row when it is not. A row of controls given less than
  its minimum width shrinks them below their size hints, which is what once
  truncated the label boxes; here a group is never squeezed, it moves.
* :class:`PagedColumn` -- a column of pages chosen from a bar of a few
  names: a page used all the time has a button, the others sit three to a
  group. It wraps a ``QTabWidget`` whose own bar is hidden, so everything
  written against the tab widget (indices, titles, ``currentChanged``)
  still holds, and the bar fits the column at every window size: ten tab
  names across it scrolled, or were cut off, below about 1700 px.
* :class:`WelcomeCard` -- what the empty window shows in place of a blank
  viewport: the three ways in.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMenu,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

GROUP_SPACING = 12
ROW_MARGINS = (10, 4, 10, 4)


# --- the toolbar ---------------------------------------------------------------------

def caption(text: str) -> QLabel:
    """The small label over a group of controls."""
    label = QLabel(text.upper())
    label.setObjectName("caption")
    return label


def group(*widgets, spacing: int = 6) -> QWidget:
    """One group of a toolbar: a row of ``widgets`` that stay together (a
    string becomes an inline label)."""
    box = QWidget()
    box.setObjectName("toolgroup")
    row = QHBoxLayout(box)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(spacing)
    for widget in widgets:
        row.addWidget(QLabel(widget) if isinstance(widget, str) else widget)
    box.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
    return box


def _rule() -> QFrame:
    line = QFrame()
    line.setObjectName("toolrule")
    line.setFixedWidth(1)
    line.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
    return line


class ReflowBar(QWidget):
    """Groups on as few rows as the width allows.

    ``trailing`` groups are pushed to the right end of the last row they
    land on. The bar's minimum width is its widest group's, so it never
    forces the window wider than one group.

    ``max_rows`` is the number of rows the bar is drawn for at the
    narrowest window, not a limit: a group that does not fit on the last of
    them starts another row. Stopping at a fixed count put what was left on
    the last row whether or not it fitted, which is the squeeze this bar
    exists to end; it showed with Qt 6.12, whose controls are a tenth wider
    than those of Qt 6.9 the count was chosen with.
    """

    def __init__(self, groups, trailing=(), max_rows: int = 3, parent=None):
        super().__init__(parent)
        self.setObjectName("toolbar")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._groups = list(groups)
        self._trailing = list(trailing)
        self._max_rows = max(1, int(max_rows))
        self._assignment: tuple | None = None
        self._column = QVBoxLayout(self)
        self._column.setContentsMargins(0, 0, 0, 0)
        self._column.setSpacing(0)
        # the bar's minimum is its own (one group), not its rows': the rows
        # are what give way, by being laid out again
        self._column.setSizeConstraint(QLayout.SetNoConstraint)
        self._rules: list[QFrame] = []
        self._rows: list[QHBoxLayout] = []
        self._holders: list[QWidget] = []
        # one row per group at the most: each group alone always fits
        for _ in range(max(self._max_rows, len(self._groups) + len(self._trailing))):
            holder = QWidget()
            row = QHBoxLayout(holder)
            row.setContentsMargins(*ROW_MARGINS)
            row.setSpacing(GROUP_SPACING)
            self._column.addWidget(holder)
            self._rows.append(row)
            self._holders.append(holder)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self._place(self._assign(10_000))

    # every group, in order, the trailing ones last
    def _all(self) -> list[QWidget]:
        return self._groups + self._trailing

    def _width_of(self, widget: QWidget) -> int:
        return widget.sizeHint().width()

    def _assign(self, width: int) -> tuple:
        """The row of each group for an available width."""
        room = width - ROW_MARGINS[0] - ROW_MARGINS[2]
        rows, used, row = [], 0, 0
        for widget in self._all():
            need = self._width_of(widget)
            if need <= 0:
                rows.append(row)
                continue
            if used and used + 2 * GROUP_SPACING + 1 + need > room:
                row, used = row + 1, 0
            used += need + (2 * GROUP_SPACING + 1 if used else 0)
            rows.append(row)
        return tuple(rows)

    def _place(self, assignment: tuple) -> None:
        if assignment == self._assignment:
            return
        self._assignment = assignment
        for row in self._rows:
            while row.count():
                row.takeAt(0)
        for line in self._rules:
            line.hide()
        first_trailing = len(self._groups)
        stretched = set()
        started: set[int] = set()
        used = 0
        for index, (widget, at) in enumerate(zip(self._all(), assignment)):
            if self._width_of(widget) <= 0:
                self._rows[at].addWidget(widget)
                continue
            if index >= first_trailing and at not in stretched:
                self._rows[at].addStretch(1)
                stretched.add(at)
            elif at in started:
                if used == len(self._rules):
                    self._rules.append(_rule())
                line = self._rules[used]
                used += 1
                self._rows[at].addWidget(line)
                line.show()
            started.add(at)
            self._rows[at].addWidget(widget, 0, Qt.AlignVCenter)
        occupied = {at for widget, at in zip(self._all(), assignment)
                    if self._width_of(widget) > 0}
        for at, row in enumerate(self._rows):
            if at not in stretched:
                row.addStretch(1)
            self._holders[at].setVisible(at in occupied)
        self.updateGeometry()

    def reflow(self) -> None:
        """Lay the groups out again (after one was shown, hidden or
        changed width)."""
        self._assignment = None
        self._place(self._assign(self.width()))

    def rows_in_use(self) -> int:
        return sum(1 for holder in self._holders if not holder.isHidden())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place(self._assign(event.size().width()))

    def minimumSizeHint(self) -> QSize:
        widest = max((self._width_of(w) for w in self._all()), default=0)
        return QSize(widest + ROW_MARGINS[0] + ROW_MARGINS[2],
                     self.sizeHint().height())

    def sizeHint(self) -> QSize:
        # the rows in use, each as tall as its tallest group
        height = 0
        for at, holder in enumerate(self._holders):
            if holder.isHidden():
                continue
            tallest = max((w.sizeHint().height() for w, row in
                           zip(self._all(), self._assignment or ()) if row == at),
                          default=0)
            height += tallest + ROW_MARGINS[1] + ROW_MARGINS[3]
        width = sum(self._width_of(w) for w in self._all())
        return QSize(max(width, 200), height)


# --- the column of pages ---------------------------------------------------------------

class PagedColumn(QWidget):
    """A ``QTabWidget`` whose pages are chosen from a bar of a few names.

    ``entries`` is a list of page titles and of ``(caption, [titles])``
    groups: a title is a button of its own, a group one button that opens
    the list of its pages and carries the name of the one on screen. Ten
    pages are then five names, which fit a 400 px column at every window
    size. The tab widget keeps its indices, titles and signals; only its
    own bar is hidden.
    """

    def __init__(self, tabs: QTabWidget, entries=(), parent=None):
        super().__init__(parent)
        self.tabs = tabs
        self._entries = [e if isinstance(e, str)
                         else (str(e[0]), [str(x) for x in e[1]])
                         for e in entries]
        self.bar = QWidget()
        self.bar.setObjectName("pagebar")
        self.bar.setAttribute(Qt.WA_StyledBackground, True)
        self._row = QHBoxLayout(self.bar)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(0)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self.bar)
        column.addWidget(tabs, 1)
        tabs.tabBar().hide()
        tabs.setDocumentMode(True)
        self.buttons: list[QToolButton] = []
        self._members: list[list[int]] = []
        self._captions: list[str] = []
        self.rebuild()
        tabs.currentChanged.connect(self._on_tab)
        tabs.tabBar().installEventFilter(self)

    def rebuild(self) -> None:
        """Make the buttons from the tab widget's titles."""
        while self._row.count():
            item = self._row.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.buttons.clear()
        self._members.clear()
        self._captions.clear()
        titles = [self.tabs.tabText(i) for i in range(self.tabs.count())]
        placed: set[int] = set()
        entries = list(self._entries) or titles
        for entry in entries:
            if isinstance(entry, str):
                if entry not in titles:
                    continue
                self._add(entry, [titles.index(entry)], titles)
            else:
                members = [titles.index(x) for x in entry[1] if x in titles]
                if members:
                    self._add(entry[0], members, titles, grouped=True)
            placed.update(self._members[-1] if self._members else ())
        for index, title in enumerate(titles):      # a page nothing names
            if index not in placed:
                self._add(title, [index], titles)
        self._row.addStretch(1)
        self._on_tab(self.tabs.currentIndex())

    def _add(self, caption: str, members: list[int], titles: list[str],
             grouped: bool = False) -> None:
        button = QToolButton()
        button.setObjectName("pagebutton")
        button.setCheckable(True)
        button.setFocusPolicy(Qt.TabFocus)
        if grouped:
            menu = QMenu(button)
            for index in members:
                action = menu.addAction(titles[index])
                action.setCheckable(True)
                action.setData(index)
                action.triggered.connect(
                    lambda _checked=False, i=index: self.tabs.setCurrentIndex(i))
            button.setMenu(menu)
            button.setPopupMode(QToolButton.InstantPopup)
        else:
            button.clicked.connect(
                lambda _checked=False, i=members[0]: self.tabs.setCurrentIndex(i))
        self._row.addWidget(button)
        self.buttons.append(button)
        self._members.append(list(members))
        self._captions.append(caption)

    def titles(self) -> list[str]:
        """Every page name the bar reaches, in the bar's order."""
        return [self.tabs.tabText(i) for members in self._members
                for i in members]

    def button_for(self, title: str) -> QToolButton:
        """The button a page is chosen from (its own, or its group's)."""
        for button, members in zip(self.buttons, self._members):
            if any(self.tabs.tabText(i) == title for i in members):
                return button
        raise KeyError(title)

    def _on_tab(self, index: int) -> None:
        for button, members, name in zip(self.buttons, self._members,
                                         self._captions):
            mine = index in members
            button.setChecked(mine)
            menu = button.menu()
            if menu is None:
                button.setText(name)
                continue
            # a group carries the name of its page on screen, else its own
            button.setText((self.tabs.tabText(index) if mine else name) + " ▾")
            for action in menu.actions():
                action.setChecked(action.data() == index)

    def fit(self) -> None:
        """Give each group button the width of its longest name, so the
        bar does not shift when a group takes the name of its page, and
        the bar's own width is the one it can ever need."""
        for button, members, name in zip(self.buttons, self._members,
                                         self._captions):
            if button.menu() is None:
                continue
            shown = button.text()
            widest = 0
            for text in [name] + [self.tabs.tabText(i) for i in members]:
                button.setText(text + " ▾")
                widest = max(widest, button.sizeHint().width())
            button.setText(shown)
            button.setMinimumWidth(widest)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.fit()                    # the style sheet's padding is known now

    def eventFilter(self, watched, event) -> bool:
        # the tab bar is hidden for good: a style change or a new tab asks
        # Qt to show it again
        if watched is self.tabs.tabBar() and event.type() == QEvent.Show:
            watched.hide()
        return False


# --- the empty window --------------------------------------------------------------------

class WelcomeCard(QFrame):
    """What the viewport shows before anything is open: the name, what the
    program does, and the ways in. It sits over the (empty) view, centred,
    and follows its size."""

    def __init__(self, host: QWidget, title: str, tagline: str, version: str,
                 actions, note: str = ""):
        super().__init__(host)
        self.setObjectName("welcome")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._host = host
        column = QVBoxLayout(self)
        column.setContentsMargins(34, 28, 34, 26)
        column.setSpacing(6)
        name = QLabel(title)
        name.setObjectName("welcomeTitle")
        column.addWidget(name)
        tag = QLabel(f"{tagline} · {version}")
        tag.setObjectName("hint")
        column.addWidget(tag)
        column.addSpacing(14)
        self.buttons: list[QPushButton] = []
        row = QHBoxLayout()
        row.setSpacing(8)
        for index, (text, target) in enumerate(actions):
            button = QPushButton(text)
            if index == 0:
                button.setObjectName("primary")
                button.setDefault(False)
            if callable(target):
                button.clicked.connect(lambda _=False, f=target: f())
            else:                               # a QMenu
                button.setMenu(target)
            row.addWidget(button)
            self.buttons.append(button)
        row.addStretch(1)
        column.addLayout(row)
        if note:
            column.addSpacing(12)
            small = QLabel(note)
            small.setObjectName("hint")
            small.setWordWrap(True)
            column.addWidget(small)
        host.installEventFilter(self)
        self._centre()

    def _centre(self) -> None:
        width = min(560, max(360, self._host.width() - 60))
        self.setFixedWidth(width)
        self.adjustSize()
        self.move(max(0, (self._host.width() - self.width()) // 2),
                  max(0, (self._host.height() - self.height()) // 2))

    def eventFilter(self, watched, event) -> bool:
        if watched is self._host and event.type() == QEvent.Resize:
            self._centre()
        return False
