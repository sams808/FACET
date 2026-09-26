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
look survives.
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
    r, g, b = rgb
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


def _muted_on(window: RGB, text: RGB, floor: float = 3.2) -> RGB:
    """Body text softened toward the panel, but only as far as stays legible.

    Secondary text has to read as secondary and still be readable, and how far
    it can be softened depends on the panel it is on. Fixing the pair instead of
    deriving it is what left the hint text at a contrast of 1.7 on a mid-grey
    ground.
    """
    best = text
    for step in range(9, -1, -1):
        candidate = _mix(text, window, step * 0.05)
        if contrast(candidate, window) >= floor:
            return candidate
        best = candidate
    return text


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
    light = _luminance(background) > 0.5

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
    muted = _muted_on(window, text)
    accent_text = _ink_on(accent)

    return UiColors(
        window=_hex(window), base=_hex(base), alternate=_hex(alternate),
        text=_hex(text), muted=_hex(muted), border=_hex(border),
        accent=_hex(accent), accent_text=_hex(accent_text), hover=_hex(hover),
        is_light=light,
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
QMenu::item {{ padding: 5px 26px 5px 24px; }}
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
                padding: 5px 10px; border: 1px solid {c.border};
                border-bottom: 0; margin-right: 1px; }}
QTabBar::tab:selected {{ background: {c.base}; color: {c.text}; }}
QTabBar::tab:hover {{ background: {c.hover}; }}
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
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: {c.base};
    color: {c.text}; border: 1px solid {c.border}; border-radius: 3px;
    padding: 2px 4px; selection-background-color: {c.accent};
    selection-color: {c.accent_text}; }}
QComboBox QAbstractItemView {{ background: {c.base}; color: {c.text};
    border: 1px solid {c.border}; selection-background-color: {c.accent};
    selection-color: {c.accent_text}; }}
QPushButton {{ background: {c.window}; color: {c.text};
    border: 1px solid {c.border}; border-radius: 3px; padding: 4px 10px; }}
QPushButton:hover {{ background: {c.hover}; }}
QPushButton:pressed {{ background: {c.accent}; color: {c.accent_text}; }}
QPushButton:disabled {{ color: {c.muted}; }}
QCheckBox, QRadioButton, QLabel {{ background: transparent; color: {c.text}; }}
QCheckBox:disabled, QRadioButton:disabled {{ color: {c.muted}; }}
QLabel#hint {{ color: {c.muted}; }}
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


def apply(target, theme=None) -> None:
    """Put ``theme``'s chrome on a QApplication (or a single widget).

    Applying to the application is what reaches menus and tool tips, which are
    top-level windows of their own and do not inherit a main window's sheet.
    """
    target.setStyleSheet(stylesheet(theme))
