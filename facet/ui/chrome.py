"""The application's own colours: menus, docks, tables, panels.

The three-dimensional view has always taken its colours from a :class:`Theme`.
The window around it did not -- it used whatever the operating system's widget
style provided, which on Windows is a light grey. That was tolerable while the
viewport was dark and read as a separate object, and wrong as soon as the
viewport became white: the picture and the frame around it were two different
whites, and a user switching to the dark preset got a dark picture in a light
window.

So the chrome is derived from the same theme. One function turns a theme into a
handful of interface roles, a second turns those into a Qt style sheet, and a
third applies both to the application. Nothing here decides *what* colour a
theme is; it only answers "given this theme, what should a menu look like".

Two deliberate choices.

*Derived, not stored.* The roles come from the theme's background luminance and
its selection colour rather than from new fields on ``Theme``, so an old theme
file saved by an earlier version still themes the whole window, and a user who
types a custom background gets a window that follows it.

*A style sheet, not a palette.* A ``QPalette`` alone does not reach menu
separators, dock titles or table grid lines under the Windows styles, and mixing
the two gives a window that is light in some places and dark in others. The
style sheet below is therefore the single source, and it is kept to colour and
spacing -- no borders redrawn, no metrics changed -- so that the platform's own
look survives. What it does not name is drawn from the palette, so the
application is given a palette of the same colours (:func:`palette`); left as
the system's, a dark system put its colours under the white theme's text.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..core import theme as theme_mod

RGB = tuple[float, float, float]


def _hex(rgb: RGB) -> str:
    r, g, b = (max(0, min(255, int(round(c * 255)))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _mix(a: RGB, b: RGB, t: float) -> RGB:
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def _luminance(rgb: RGB) -> float:
    """Relative luminance, with the sRGB transfer function undone first.

    The gamma step is not optional. Skipping it -- weighting the raw 0..1
    channel values -- reports a highlighted menu row in the High contrast theme
    at 7.15 where the screen renders it at 3.30, because a saturated mid-tone
    blue is far darker in light than its channel values suggest. Every contrast
    decision here rests on this function, so it has to be the real one.
    """
    channels = []
    for c in rgb:
        c = max(0.0, min(1.0, float(c)))
        channels.append(c / 12.92 if c <= 0.04045
                        else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: RGB, b: RGB) -> float:
    """How far apart two colours are, in the WCAG contrast-ratio form.

    Used to choose ink rather than to grade a design, so the channel values go
    in as they are instead of being linearised first; the ordering is the same
    either way and that is all this decides.
    """
    lo, hi = sorted((_luminance(a), _luminance(b)))
    return (hi + 0.05) / (lo + 0.05)


def _settle(rgb: RGB, target: RGB, limit: float, above: bool) -> RGB:
    """Push ``rgb`` toward ``target`` until its luminance passes ``limit``.

    Used to keep the panels away from mid-grey. The viewport may be any colour a
    theme likes -- a mid-grey ground is a deliberate choice for looking at
    structures -- but a panel at mid-grey can carry neither dark nor light text,
    so the frame steps away from the middle even when the picture does not.
    """
    out = rgb
    for _ in range(16):
        lum = _luminance(out)
        if (above and lum >= limit) or (not above and lum <= limit):
            break
        out = _mix(out, target, 0.2)
    return out


def _as_drawn(rgb: RGB) -> RGB:
    """``rgb`` rounded to the 8-bit channels the ``#rrggbb`` string carries,
    so that a contrast checked here is the contrast of what is drawn."""
    return tuple(max(0, min(255, int(round(c * 255)))) / 255 for c in rgb)


def _floor_on(rgb: RGB, grounds, floor: float) -> bool:
    drawn = _as_drawn(rgb)
    return all(contrast(drawn, _as_drawn(g)) >= floor for g in grounds)


# Secondary text -- hints, group titles, table headers, unselected tabs,
# disabled controls and menu items, placeholders -- is held to the same 4.5:1
# as body text. It used to be softened to 3.2:1, which read as secondary but
# measured 3.80 on the white theme's panels and 3.36 for a placeholder on a
# white field; it still reads as secondary at 4.5, beside body text at 15.
MUTED_FLOOR = 4.5


def _muted_on(window: RGB, text: RGB, floor: float = MUTED_FLOOR,
              grounds=()) -> RGB:
    """Body text softened toward the panel, but only as far as stays legible.

    Secondary text has to read as secondary and still be readable, and how far
    it can be softened depends on the panel it is on. Fixing the pair instead of
    deriving it is what left the hint text at a contrast of 1.7 on a mid-grey
    ground. ``grounds``: the other backgrounds the same colour is drawn on
    (fields, alternate rows, menus), every one of which must keep ``floor``.
    """
    grounds = (window,) + tuple(grounds)
    for step in range(50, -1, -1):
        candidate = _mix(text, window, step * 0.01)
        if _floor_on(candidate, grounds, floor):
            return candidate
    return text


def _link_on(light: bool, grounds, text: RGB, floor: float = 4.5) -> RGB:
    """A link colour that reads as a link and carries on every ground.

    Links took the platform's colour: (233, 212, 242) from a dark system
    palette on the white theme's panels measured 1.23. A blue, pushed toward
    the body text until it carries.
    """
    out: RGB = (0.04, 0.33, 0.78) if light else (0.55, 0.74, 1.0)
    for _ in range(24):
        if _floor_on(out, grounds, floor):
            break
        out = _mix(out, text, 0.15)
    return out


def _ink_on(background: RGB) -> RGB:
    """Black or white, whichever carries further on ``background``.

    A fixed lightness threshold gets a mid-tone wrong: the shipped orange
    selection colour sits close enough to the middle that black on it reads
    better than white, and any threshold at 0.5 or 0.6 chooses white.
    """
    # Pure black rather than the body ink: this is a small label on a
    # saturated patch, where every bit of separation counts.
    white, black = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
    if contrast(white, background) > contrast(black, background):
        return white
    return black


@dataclass(frozen=True)
class UiColors:
    """The interface roles a window needs, as ``#rrggbb`` strings."""

    window: str        # panels, menu bars, dock titles
    base: str          # text areas, tables, editable fields
    alternate: str     # alternating table rows
    text: str          # body text
    muted: str         # secondary text: hints, notes, units
    border: str        # separators and frames
    accent: str        # selection and focus
    accent_text: str   # text drawn on the accent
    hover: str         # menu and row hover
    is_light: bool
    link: str = "#0a54c7"   # links in labels and the manual

    def as_dict(self) -> dict[str, str]:
        return {k: v for k, v in self.__dict__.items() if isinstance(v, str)}


def ui_colors(theme=None) -> UiColors:
    """Interface colours for ``theme`` (the default theme when None).

    The window is not the viewport's colour but a step away from it: a panel
    that is exactly the same white as the picture makes the picture's edges
    disappear, and a group box drawn on it has nothing to sit on.
    """
    background: RGB = tuple(getattr(theme, "background", None)
                            or theme_mod.FALLBACK_BACKGROUND)
    accent: RGB = tuple(getattr(theme, "selection_color", None)
                        or theme_mod.FALLBACK_SELECTION_COLOR)
    # Against the linearised luminance the midpoint is not 0.5: sRGB 50% grey
    # linearises to 0.21, and a white-background theme must still read as light.
    light = _luminance(background) > 0.18

    if light:
        window = _settle(_mix(background, (0.0, 0.0, 0.0), 0.055),
                         (1.0, 1.0, 1.0), 0.82, above=True)
        base = (1.0, 1.0, 1.0)
        alternate = _mix(base, (0.0, 0.0, 0.0), 0.032)
        text = (0.09, 0.10, 0.12)
        border = _mix(window, (0.0, 0.0, 0.0), 0.16)
        hover = _mix(window, (0.0, 0.0, 0.0), 0.07)
    else:
        window = _settle(_mix(background, (1.0, 1.0, 1.0), 0.07),
                         (0.0, 0.0, 0.0), 0.16, above=False)
        base = _mix(window, (1.0, 1.0, 1.0), 0.03)
        alternate = _mix(base, (1.0, 1.0, 1.0), 0.035)
        text = (0.90, 0.92, 0.95)
        border = _mix(window, (1.0, 1.0, 1.0), 0.16)
        hover = _mix(window, (1.0, 1.0, 1.0), 0.10)
    # secondary text sits on the panels, in fields and tables, and in menus
    grounds = (base, alternate)
    muted = _muted_on(window, text, grounds=grounds)
    accent_text = _ink_on(accent)
    link = _link_on(light, (window,) + grounds, text)

    return UiColors(
        window=_hex(window), base=_hex(base), alternate=_hex(alternate),
        text=_hex(text), muted=_hex(muted), border=_hex(border),
        accent=_hex(accent), accent_text=_hex(accent_text), hover=_hex(hover),
        is_light=light, link=_hex(link),
    )


def muted_hex(theme=None) -> str:
    """The colour of secondary text, for building HTML by hand."""
    return ui_colors(theme).muted


def text_hex(theme=None) -> str:
    return ui_colors(theme).text


def stylesheet(theme=None) -> str:
    """A Qt style sheet putting ``theme``'s colours on the whole window.

    ``QLabel#hint`` is the one piece of naming this relies on: every panel's
    explanatory small print carries that object name, so the hint colour lives
    here rather than in seventeen separate calls to ``setStyleSheet``.
    """
    c = ui_colors(theme)
    return f"""
QWidget {{ color: {c.text}; }}
QMainWindow, QDialog, QWidget#panel {{ background: {c.window}; }}
QMenuBar {{ background: {c.window}; color: {c.text}; border: 0; }}
QMenuBar::item {{ background: transparent; padding: 4px 9px; }}
QMenuBar::item:selected {{ background: {c.hover}; }}
QMenuBar::item:pressed {{ background: {c.accent}; color: {c.accent_text}; }}
QMenu {{ background: {c.base}; color: {c.text};
         border: 1px solid {c.border}; padding: 4px; }}
/* No FACET action carries an icon, so the usual icon gutter is dead space;
   it also pushed the label 27 px away from the tick on a checkable row. */
QMenu::item {{ padding: 5px 18px 5px 10px; }}
QMenu::item:selected {{ background: {c.accent}; color: {c.accent_text}; }}
QMenu::item:disabled {{ color: {c.muted}; }}
QMenu::separator {{ height: 1px; background: {c.border}; margin: 4px 8px; }}
QMenu::indicator {{ width: 14px; }}
QToolTip {{ background: {c.base}; color: {c.text};
            border: 1px solid {c.border}; padding: 3px; }}
QStatusBar {{ background: {c.window}; color: {c.muted}; }}
QStatusBar::item {{ border: 0; }}
QDockWidget {{ color: {c.text}; }}
QDockWidget::title {{ background: {c.window}; color: {c.muted};
                      padding: 5px 8px; border-bottom: 1px solid {c.border}; }}
QGroupBox {{ background: transparent; border: 1px solid {c.border};
             border-radius: 4px; margin-top: 10px; padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px;
                    color: {c.muted}; }}
QTabWidget::pane {{ background: {c.window}; border: 1px solid {c.border}; }}
QTabBar {{ background: transparent; }}
QTabBar::tab {{ background: {c.window}; color: {c.muted};
                padding: 5px 5px; border: 1px solid {c.border};
                border-bottom: 0; margin-right: 1px; }}
QTabBar::tab:selected {{ background: {c.base}; color: {c.text}; }}
/* body text on hover: the secondary colour is held to 4.5:1 on the panels,
   not on the darker hover fill */
QTabBar::tab:hover {{ background: {c.hover}; color: {c.text}; }}
QTableWidget, QTableView, QTreeView, QListWidget, QListView, QTextBrowser,
QTextEdit, QPlainTextEdit {{ background: {c.base}; color: {c.text};
    alternate-background-color: {c.alternate};
    selection-background-color: {c.accent}; selection-color: {c.accent_text};
    border: 1px solid {c.border}; }}
QHeaderView {{ background: {c.window}; }}
QHeaderView::section {{ background: {c.window}; color: {c.muted};
    padding: 3px 6px; border: 0; border-right: 1px solid {c.border};
    border-bottom: 1px solid {c.border}; }}
QTableWidget::item:selected, QTableView::item:selected,
QListWidget::item:selected, QTreeView::item:selected {{
    background: {c.accent}; color: {c.accent_text}; }}
QTableCornerButton::section {{ background: {c.window};
                               border: 0; border-bottom: 1px solid {c.border}; }}
QLineEdit, QComboBox {{ background: {c.base};
    color: {c.text}; border: 1px solid {c.border}; border-radius: 3px;
    padding: 2px 4px; selection-background-color: {c.accent};
    selection-color: {c.accent_text}; placeholder-text-color: {c.muted}; }}
/* No border on a spin box: a border hands its up and down buttons from the
   Windows 11 style to the old bevelled drawing, whose arrows came out as two
   dots. The padding is what keeps the size the panels were laid out with. */
QSpinBox, QDoubleSpinBox {{ background: {c.base}; color: {c.text};
    padding: 2px 4px; selection-background-color: {c.accent};
    selection-color: {c.accent_text}; }}
QTextEdit, QPlainTextEdit {{ placeholder-text-color: {c.muted}; }}
QComboBox QAbstractItemView {{ background: {c.base}; color: {c.text};
    border: 1px solid {c.border}; selection-background-color: {c.accent};
    selection-color: {c.accent_text}; }}
QPushButton {{ background: {c.window}; color: {c.text};
    border: 1px solid {c.border}; border-radius: 3px; padding: 4px 10px; }}
QPushButton:hover {{ background: {c.hover}; }}
QPushButton:pressed, QPushButton:checked {{ background: {c.accent};
    color: {c.accent_text}; }}
QPushButton:disabled {{ color: {c.muted}; }}
/* A ticked tool button (the results' Provenance) is filled with the accent;
   its text has to be the accent's ink, not the body text: dark text on the
   accent measured 2.87. */
QToolButton:checked {{ background: {c.accent}; color: {c.accent_text}; }}
/* No background on a check box or a radio button: under the Windows 11 style
   'transparent' also took the accent fill out of a ticked box, leaving a
   white tick on the panel (1.11:1) -- ticked and unticked looked alike. */
QCheckBox, QRadioButton {{ color: {c.text}; }}
QLabel {{ background: transparent; color: {c.text}; }}
QCheckBox:disabled, QRadioButton:disabled {{ color: {c.muted}; }}
QLabel#hint {{ color: {c.muted}; }}
QToolButton#disclosure {{ background: transparent; border: 0; color: {c.muted};
                          padding: 2px 0; }}
QToolButton#disclosure:hover {{ color: {c.text}; }}
QSlider::groove:horizontal {{ height: 4px; background: {c.border};
                              border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {c.accent}; width: 12px;
    margin: -5px 0; border-radius: 6px; }}
QProgressBar {{ background: {c.base}; color: {c.text};
    border: 1px solid {c.border}; border-radius: 3px; text-align: center; }}
QProgressBar::chunk {{ background: {c.accent}; }}
QScrollBar:vertical {{ background: {c.window}; width: 11px; margin: 0; }}
QScrollBar:horizontal {{ background: {c.window}; height: 11px; margin: 0; }}
QScrollBar::handle {{ background: {c.border}; border-radius: 5px;
                      min-height: 24px; min-width: 24px; }}
QScrollBar::handle:hover {{ background: {c.muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QSplitter::handle {{ background: {c.border}; }}
QSplitter::handle:horizontal {{ width: 2px; }}
QSplitter::handle:vertical {{ height: 2px; }}
""".strip()


def mark_hint(widget) -> None:
    """Tag a label as secondary text, so the style sheet colours it."""
    widget.setObjectName("hint")


# ---------------------------------------------------------------------------
# laying controls out in a narrow panel
# ---------------------------------------------------------------------------
# The right-hand panels get 398 px, and a row of ten label-and-field pairs asks
# for four times that. Qt resolves the shortfall by shrinking children below
# their minimum size hint, which does not degrade gracefully: measured, twelve
# labels on the Diffraction tab end up exactly zero pixels wide -- not clipped,
# absent -- and two combo boxes get zero and minus seven pixels of text area.
#
# So the panels do not use separate label widgets. A spin box states its own
# name in its prefix and its unit in its suffix, and a combo box already shows
# its current item; what is left is a grid of self-describing controls, two per
# row, with the wide ones spanning. The sentence that used to be a label goes
# into the tool tip, where there is room for it.

FIELD_SPACING = 6


def fit_combo(box) -> None:
    """Stop a combo box and its popup from eliding their own items.

    Call once, after the items are added. Two separate mechanisms are needed.
    ``AdjustToContents`` makes the *popup* size itself from the widest item
    rather than from the closed combo, which may have been squeezed; and the
    popup's view elides in the middle under the Windows styles, which is what
    turned "bond valence" into "bon...nce". ``setMinimumWidth`` is then the only
    thing a layout that has been given less room than it asked for still
    honours, so it is what keeps the *closed* combo readable.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QComboBox

    box.setSizeAdjustPolicy(QComboBox.AdjustToContents)
    view = box.view()
    if view is not None:
        view.setTextElideMode(Qt.ElideNone)
    box.setMinimumWidth(box.sizeHint().width())


def in_scroll_area(widget):
    """Wrap a panel so it scrolls rather than forcing the window taller.

    ``setWidgetResizable(True)`` is what makes this free: when the page has
    room it is resized to the viewport and behaves exactly as an unwrapped
    panel, and only when it does not does a scroll bar appear. Without it the
    window's minimum height was 1421 px on a 930 px screen -- the Volume page
    demanding 1130 px of it -- so the window could not be made to fit the
    display.

    The page and the scroll area's viewport are both left unfilled, so what
    shows behind the page is the window or tab pane the area sits in, which
    the style sheet colours from the theme. ``setWidget`` turns the page's
    ``autoFillBackground`` on, and the viewport fills by default; under an
    application style sheet a widget inherits no palette from its parent, so
    both filled with the operating system's Window colour rather than the
    theme's. Measured with Windows in dark mode: the theme's dark text on
    (30, 30, 30), a contrast of 1.05, on the scrolling pages of both windows;
    with Windows light and the Dark theme, light text on (243, 243, 243), 1.08.
    A style-sheet rule cannot do this: the viewport is not a widget a style
    sheet styles, and a background given to a page that is a QWidget subclass
    -- every FACET panel -- is never painted.
    """
    from PySide6.QtWidgets import QScrollArea

    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.NoFrame)
    area.setWidget(widget)
    widget.setAutoFillBackground(False)
    area.viewport().setAutoFillBackground(False)
    # the panel is what callers hold on to; keep it reachable from the wrapper
    area.panel = widget
    return area


def fit_every_combo(widget) -> None:
    """Apply ``fit_combo`` to every combo box under ``widget``.

    A sweep rather than a call per box, because the per-box version was added
    once and then missed on every panel written afterwards: measured, 8 of 17
    panel drop-downs could not show their own longest item and 6 could not show
    even their current text. Applying it to a whole panel cannot be forgotten
    when a control is added.
    """
    from PySide6.QtWidgets import QComboBox

    for box in widget.findChildren(QComboBox):
        fit_combo(box)


def name_inside(widget, name: str = "", unit: str = "", *, tip: str = ""):
    """Move a control's label into the control, and return the control.

    For a spin box the name becomes the prefix and the unit the suffix. Qt keeps
    both out of the way of editing: selecting all inside the box selects only
    the number, and ``valueFromText`` strips the prefix before parsing, so
    typing a value works exactly as it did with a separate label.

    For a combo box there is nothing to prefix -- it shows its current item --
    so this only stops it eliding and gives it the tool tip.

    Read the value with ``value()`` as before. ``text()`` now includes the
    prefix; ``cleanText()`` is the one that does not.
    """
    from PySide6.QtWidgets import QAbstractSpinBox, QComboBox, QSizePolicy

    if isinstance(widget, QAbstractSpinBox):
        if name:
            widget.setPrefix(f"{name}  ")
        if unit:
            widget.setSuffix(unit)
    elif isinstance(widget, QComboBox):
        fit_combo(widget)
    if tip:
        widget.setToolTip(tip)
    if name:
        # what a screen reader and Qt's own test tooling read, since there is
        # no longer a QLabel saying it
        widget.setAccessibleName(name)
    widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
    return widget


def field_grid(items, columns: int = 2, spacing: int = FIELD_SPACING):
    """Pack self-labelled controls into a grid that fits a narrow panel.

    ``items`` is a sequence of widgets, or of ``(widget, span)`` pairs where a
    span of ``columns`` gives the widget a whole row. A bare ``None`` starts a
    new row, for grouping.

    Returns the ``QGridLayout``, so the caller decides what to put it in.
    """
    from PySide6.QtWidgets import QGridLayout

    grid = QGridLayout()
    grid.setHorizontalSpacing(spacing)
    grid.setVerticalSpacing(spacing)
    grid.setContentsMargins(0, 0, 0, 0)
    for column in range(columns):
        grid.setColumnStretch(column, 1)

    row, column = 0, 0
    for item in items:
        if item is None:
            if column:
                row, column = row + 1, 0
            continue
        widget, span = item if isinstance(item, tuple) else (item, 1)
        span = max(1, min(int(span), columns))
        if column + span > columns:
            row, column = row + 1, 0
        grid.addWidget(widget, row, column, 1, span)
        column += span
        if column >= columns:
            row, column = row + 1, 0
    return grid


def disclosure(title: str, inner, *, expanded: bool = False):
    """A button that shows or hides ``inner``, and the pair as one widget.

    For the settings that are set once and then left. Hiding them is what lets
    the controls people actually reach for stay visible in a panel this narrow;
    the alternative -- a toolbar's overflow button -- was measured to hide nine
    of eleven controls, and which nine is decided by the order in the source
    rather than by how often they are wanted.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QToolButton, QVBoxLayout, QWidget

    holder = QWidget()
    column = QVBoxLayout(holder)
    column.setContentsMargins(0, 0, 0, 0)
    column.setSpacing(FIELD_SPACING)

    button = QToolButton()
    button.setText(title)
    button.setCheckable(True)
    button.setChecked(bool(expanded))
    button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
    button.setAutoRaise(True)
    button.setObjectName("disclosure")
    column.addWidget(button)
    column.addWidget(inner)
    inner.setVisible(bool(expanded))

    def toggled(on):
        inner.setVisible(bool(on))
        button.setArrowType(Qt.DownArrow if on else Qt.RightArrow)

    button.toggled.connect(toggled)
    holder.button = button
    holder.inner = inner
    return holder


def match_color_scheme(theme=None) -> bool:
    """Ask the platform for the light or the dark look, as ``theme`` is.

    The style sheet colours what it names. What it leaves to the widget
    style -- a check box's or a list item's tick box, a radio button, a tool
    button's face, a combo box's arrow, a spin box's buttons -- the Windows 11
    style draws from the *system's* colour scheme, whatever the palette says.
    Measured with Windows in dark mode and the white theme: unticked boxes all
    but invisible on the light panels, and the results' Provenance button a
    dark face under the theme's dark text. Asking for the theme's scheme
    makes those parts agree with the rest of the window; with Windows already
    in that scheme nothing changes.

    Returns True when Qt can be asked (6.8 and later; a platform may still
    ignore the request), False when it cannot.

    The scheme is asked for every time, even when the system's already
    matches: an explicit request is what holds it, so that switching Windows
    between light and dark while FACET runs does not bring the mismatch back.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication

    hints = QGuiApplication.styleHints()
    request = getattr(hints, "setColorScheme", None)
    if request is None:
        return False
    wanted = (Qt.ColorScheme.Light if ui_colors(theme).is_light
              else Qt.ColorScheme.Dark)
    request(wanted)
    return True


def palette(theme=None):
    """A ``QPalette`` carrying the style sheet's colours.

    The style sheet colours what it names; everything else a widget draws
    -- a link in a label, a placeholder, a tool button's face, a tick box's
    fill, a scroll area's page -- comes from the palette, and the
    application's palette is the system's. With Windows dark, or any platform
    that leaves its own palette in place, that put the system's colours
    under the theme's: links at (233, 212, 242) on the white theme's panels
    (1.23:1), a tool button's (60, 60, 60) face under dark text (1.58), the
    accent of the operating system rather than the theme's behind a ticked
    button. Giving the application this palette makes the two agree, which
    is what the module's note warns mixing them does not do when they differ.
    """
    from PySide6.QtGui import QColor, QPalette

    c = ui_colors(theme)

    def rgb(hexcolor: str) -> RGB:
        return tuple(int(hexcolor[i:i + 2], 16) / 255 for i in (1, 3, 5))

    window = rgb(c.window)
    black, white = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    roles = {
        "Window": c.window, "WindowText": c.text, "Base": c.base,
        "AlternateBase": c.alternate, "Text": c.text, "Button": c.window,
        "ButtonText": c.text, "BrightText": c.base,
        "ToolTipBase": c.base, "ToolTipText": c.text,
        "Highlight": c.accent, "HighlightedText": c.accent_text,
        "Link": c.link, "LinkVisited": c.link, "PlaceholderText": c.muted,
        "Accent": c.accent,
        # the bevel shades, lighter and darker than a button's face
        "Light": _hex(_mix(window, white, 0.6)),
        "Midlight": _hex(_mix(window, white, 0.3)),
        "Mid": _hex(_mix(window, black, 0.18)),
        "Dark": _hex(_mix(window, black, 0.35)),
        "Shadow": _hex(_mix(window, black, 0.7)),
    }
    pal = QPalette()
    for name, colour in roles.items():
        role = getattr(QPalette, name, None)     # Accent: Qt 6.6 and later
        if role is None:
            continue
        for group in (QPalette.Active, QPalette.Inactive, QPalette.Disabled):
            pal.setColor(group, role, QColor(colour))
    for name in ("WindowText", "Text", "ButtonText"):
        pal.setColor(QPalette.Disabled, getattr(QPalette, name),
                     QColor(c.muted))
    return pal


def apply(target, theme=None) -> None:
    """Put ``theme``'s chrome on a QApplication (or a single widget).

    Applying to the application is what reaches menus and tool tips, which are
    top-level windows of their own and do not inherit a main window's sheet.
    It also asks the platform for the theme's light or dark scheme
    (:func:`match_color_scheme`) and gives the application the palette of
    the same colours (:func:`palette`), which a single widget cannot.
    """
    from PySide6.QtWidgets import QApplication

    if isinstance(target, QApplication):
        match_color_scheme(theme)
        wanted = palette(theme)
        if target.palette() != wanted:
            target.setPalette(wanted)
    target.setStyleSheet(stylesheet(theme))
