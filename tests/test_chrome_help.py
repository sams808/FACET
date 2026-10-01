"""The window's own colours, the About dialog, and the manual.

Three things are checked here that a green suite had no way of noticing before:
that the shipped theme is the white one it is documented to be, that changing
the default did not silently redefine another preset, and that every colour a
panel draws by hand follows the theme rather than a literal chosen when the
application was dark.
"""
from __future__ import annotations

import re

import pytest

from conftest import dispose

from facet.core import theme as T
from facet.ui import chrome


# --- the derived interface colours ------------------------------------------

def test_the_default_theme_is_a_white_ground():
    """The shipped default, as documented in the manual and the README."""
    t = T.Theme()
    assert t.is_light_background
    assert t.background == (1.0, 1.0, 1.0)
    assert chrome.ui_colors(t).is_light


def test_the_preset_and_the_dataclass_defaults_are_one_statement():
    """``vesta()`` spells out what ``Theme()`` defaults to.

    The trap this guards: ``dark()`` used to be a bare ``Theme(name="Dark")``,
    so the dark values *were* the defaults and making the default white would
    have rewritten the Dark preset without touching it.
    """
    assert T.vesta().to_dict() == T.Theme().to_dict() | {"name": "VESTA-like (white)"}


def test_the_dark_preset_is_still_dark():
    assert not T.dark().is_light_background
    assert not T.slate().is_light_background
    for name in ("VESTA-like (white)", "Light (for print)",
                 "Publication (greyscale)", "High contrast"):
        assert T.PRESETS[name]().is_light_background, name


def test_every_preset_states_every_colour_itself():
    """No preset may inherit a colour from the default theme.

    A preset that leaves a field out follows whatever the default becomes, which
    is how a dark preset ends up with a white unit cell.
    """
    fields = ("background", "cell_color", "label_color", "selection_color",
              "polyhedron_color", "subthreshold_color", "uniform_atom_color",
              "uniform_bond_color")
    import inspect

    for name, factory in T.PRESETS.items():
        source = inspect.getsource(factory)
        for field in fields:
            assert f"{field}=" in source, f"{name} does not state {field}"


def test_the_fallback_constants_match_the_default_theme():
    """The renderers draw before a theme exists; the two must not disagree."""
    t = T.Theme()
    assert T.FALLBACK_BACKGROUND == t.background
    assert T.FALLBACK_CELL_COLOR == t.cell_color
    assert T.FALLBACK_LABEL_COLOR == t.label_color
    assert T.FALLBACK_SELECTION_COLOR == t.selection_color
    assert T.FALLBACK_FOG == t.fog_amount


def test_a_renderer_built_with_no_theme_uses_the_fallback(qapp):
    from facet.gl.painter import PainterRenderer

    r = PainterRenderer()
    assert r.background.getRgb()[:3] == tuple(
        round(c * 255) for c in T.FALLBACK_BACKGROUND)


def test_the_chrome_follows_the_theme_both_ways():
    light = chrome.ui_colors(T.vesta())
    dark = chrome.ui_colors(T.dark())
    assert light.is_light and not dark.is_light
    # body text must be dark on the light theme and light on the dark one
    assert int(light.text[1:3], 16) < 128 < int(dark.text[1:3], 16)
    # and the panels must not be the same colour as the viewport, or the
    # viewport's edges vanish
    assert light.window != "#ffffff"


def _rgb(hexcolor):
    """A #rrggbb string as the 0..1 triple chrome works in."""
    return tuple(int(hexcolor[i:i + 2], 16) / 255 for i in (1, 3, 5))


def test_secondary_text_is_legible_against_its_panel():
    """The hint colour has to carry against the window it is drawn on.

    Measured with ``chrome.contrast``, which undoes the sRGB transfer function
    first. A test with its own naive copy of the formula agreed with the code
    only because the code shared the same mistake, and between them they scored
    a menu row at 7.15 that the screen renders at 3.30.
    """
    for factory in T.PRESETS.values():
        c = chrome.ui_colors(factory())
        ratio = chrome.contrast(_rgb(c.muted), _rgb(c.window))
        assert ratio > 3.0, (factory().name, ratio)


def test_accent_text_is_legible_on_the_accent():
    """A selected menu row and a selected table row are drawn like this."""
    for factory in T.PRESETS.values():
        t = factory()
        c = chrome.ui_colors(t)
        ratio = chrome.contrast(_rgb(c.accent), _rgb(c.accent_text))
        assert ratio > 4.5, (t.name, ratio)


def test_the_luminance_undoes_the_srgb_transfer_function():
    """Against the published sRGB relative-luminance values.

    Mid grey is the case that matters: sRGB 50% has a relative luminance of
    0.2140, not 0.5, and treating it as 0.5 is what let a dark saturated blue
    pass for a light background.
    """
    assert chrome._luminance((0.0, 0.0, 0.0)) == pytest.approx(0.0)
    assert chrome._luminance((1.0, 1.0, 1.0)) == pytest.approx(1.0)
    assert chrome._luminance((0.5, 0.5, 0.5)) == pytest.approx(0.2140, abs=1e-3)
    # the primaries, at their published weights
    assert chrome._luminance((1.0, 0.0, 0.0)) == pytest.approx(0.2126, abs=1e-4)
    assert chrome._luminance((0.0, 1.0, 0.0)) == pytest.approx(0.7152, abs=1e-4)
    assert chrome._luminance((0.0, 0.0, 1.0)) == pytest.approx(0.0722, abs=1e-4)
    # and black against white is the textbook 21:1
    assert chrome.contrast((0.0, 0.0, 0.0),
                           (1.0, 1.0, 1.0)) == pytest.approx(21.0, abs=1e-6)


def test_the_stylesheet_reaches_menus_and_tables():
    """A palette alone does not colour these, which is why a sheet is used."""
    sheet = chrome.stylesheet(T.Theme())
    for selector in ("QMenuBar", "QMenu::item:selected", "QDockWidget::title",
                     "QHeaderView::section", "QTabBar::tab:selected",
                     "QLabel#hint", "QToolTip", "QComboBox QAbstractItemView"):
        assert selector in sheet, selector
    # no colour may be left as a literal from the dark era
    assert "8a93a3" not in sheet


def test_no_panel_still_hard_codes_the_old_hint_colour():
    """Every hint now carries an object name instead of its own sheet."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "facet"
    offenders = [
        str(f.relative_to(root)) for f in root.rglob("*.py")
        if "8a93a3" in f.read_text(encoding="utf-8")
        and f.name not in ("branding.py", "cutoff_explorer.py")
    ]
    assert not offenders, offenders


# --- the dialogs ------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_the_about_text_states_the_version_and_the_model(qapp):
    from facet.core import bv
    from facet.ui.help import about_html
    from facet.version import __version__

    html = about_html("FULL (OpenGL 3.3)", T.Theme())
    assert __version__ in html
    assert "sams808/FACET" in html
    assert f"{bv.DEFAULT.b}" in html               # b, from the code
    assert "FULL (OpenGL 3.3)" in html             # the tier actually in use
    assert bv.DEFAULT.name in html                 # the parameter set in use
    assert "LGPL" in html                          # the redistribution condition


def test_the_about_dialog_carries_the_mark(qapp):
    from facet.ui.help import AboutDialog

    d = AboutDialog(None, renderer="software", theme=T.Theme())
    try:
        assert "FACET" in d.windowTitle()
        assert d.browser.toPlainText().strip()
        mark = d.layout().itemAt(0).widget()
        assert not mark.pixmap().isNull()
    finally:
        d.deleteLater()


def test_the_manual_has_every_section_it_is_addressed_by(qapp):
    from facet.ui.help import ManualDialog, _sections

    keys = [k for k, _t, _b in _sections()]
    assert len(set(keys)) == len(keys), "duplicate section key"
    # the keys other code names
    for key in ("start", "why", "shortcuts", "licences", "lonepair",
                "diffraction", "exafs", "limits"):
        assert key in keys, key

    d = ManualDialog(None, section="shortcuts")
    try:
        assert d.contents.count() == len(keys)
        assert "Keyboard" in d.contents.currentItem().text()
        d.show_section("licences")
        assert "Licence" in d.contents.currentItem().text()
        # an unknown key opens the first section rather than raising
        d.show_section("no-such-section")
        assert d.contents.currentRow() == 0
        d._show_everything()
        assert len(d.browser.toPlainText()) > 10000
    finally:
        d.deleteLater()


def test_the_manual_states_the_thresholds_from_the_code(qapp):
    """A manual with its own copy of the numbers drifts from the program."""
    from facet.core import bv
    from facet.ui.help import _offset, manual_html

    html = manual_html()
    assert f"{bv.V_BOND_DEFAULT:g}" in html
    assert f"{bv.V_LIST_DEFAULT:g}" in html
    assert f"{_offset(bv.V_BOND_DEFAULT):.3f}" in html
    # and the offset is the inverse of the valence expression, not a constant
    assert _offset(bv.V_BOND_DEFAULT) == pytest.approx(0.9584, abs=5e-4)


def test_the_manual_says_what_is_not_done(qapp):
    """The limits section is the one that keeps the tool from being misused."""
    from facet.ui.help import _sections

    limits = next(b for k, _t, b in _sections() if k == "limits").lower()
    for claim in ("does not refine", "does not judge", "xanes",
                  "ships one bond-valence table, not a compilation"):
        assert claim in limits, claim


def test_the_manual_never_calls_a_number_good_or_bad(qapp):
    """The same rule the panels are held to, applied to the documentation.

    'should' is allowed nowhere near a measurement, so the whole manual is
    checked rather than only the parts that quote numbers.
    """
    from facet.ui.help import _sections

    forbidden = {"good", "bad", "poor", "excellent", "reliable",
                 "unreliable", "trustworthy", "untrustworthy"}
    for key, _title, body in _sections():
        text = re.sub(r"<[^>]+>", " ", body).lower()
        words = set(re.findall(r"[a-z]+", text))
        hits = words & forbidden
        # the limits section names the words it declines to use, deliberately
        if key == "limits":
            continue
        assert not hits, (key, hits)


# --- the window -------------------------------------------------------------

@pytest.fixture
def window(qapp):
    from facet.ui.preview import PreviewWindow

    w = PreviewWindow()
    w.show()
    yield w
    dispose(w)


def test_every_theme_in_the_menu_can_be_chosen(window, qapp):
    from PySide6.QtWidgets import QApplication

    for name in T.PRESETS:
        window._choose_theme(name)
        qapp.processEvents()
        assert window.theme.name == name
        ticked = [n for n, a in window._theme_actions.items() if a.isChecked()]
        assert ticked == [name], (name, ticked)
        assert QApplication.instance().styleSheet(), "the chrome was not applied"


def test_a_theme_edited_by_hand_ticks_nothing(window):
    """Half a preset is not a preset, and pretending otherwise misleads."""
    window._choose_theme("Dark")
    window.theme.name = "Mine"
    window._sync_theme_menu(window.theme)
    assert not any(a.isChecked() for a in window._theme_actions.values())


def test_the_label_combo_boxes_can_show_their_widest_item(window, qapp):
    """The bug in the screenshot: 'bond valence' rendered as 'bon...nce'.

    The mechanism was a single toolbar row asking for about 1400 px in a 900 px
    column -- Qt then shrinks children *below* their minimum size hint, and the
    popup inherits the squeezed width.
    """
    from PySide6.QtCore import Qt

    for width in (1000, 1400, 1920):
        window.resize(width, 900)
        qapp.processEvents()
        for name in ("style_box", "poly_box", "atom_label_box",
                     "bond_label_box"):
            box = getattr(window, name)
            needed = max(box.fontMetrics().horizontalAdvance(box.itemText(i))
                         for i in range(box.count()))
            assert box.width() >= needed, (name, width, box.width(), needed)
            assert box.view().textElideMode() == Qt.ElideNone, name


def test_the_label_items_still_carry_their_full_names(window):
    """The cheap fix would have been to shorten these; exports quote them."""
    from facet.gl.labels import AtomLabel, BondLabel

    texts = {window.bond_label_box.itemText(i)
             for i in range(window.bond_label_box.count())}
    assert {k.value for k in BondLabel} == texts
    texts = {window.atom_label_box.itemText(i)
             for i in range(window.atom_label_box.count())}
    assert {k.value for k in AtomLabel} == texts


def test_switching_theme_rewrites_the_hand_built_html(window, qapp):
    """The Site tab's HTML is written by the window, not styled by Qt."""
    window._choose_theme("Dark")
    qapp.processEvents()
    dark_muted = chrome.muted_hex(window.theme)
    window._choose_theme("VESTA-like (white)")
    qapp.processEvents()
    light_muted = chrome.muted_hex(window.theme)
    assert dark_muted != light_muted
    # the placeholder is rebuilt with the new colour
    assert light_muted.lstrip("#") in window.analysis.toHtml().lower()


def test_no_menu_has_two_items_on_the_same_mnemonic(window):
    """Two items in one menu claiming the same letter breaks the keyboard.

    Qt does not complain: it moves the highlight to the next match instead of
    triggering, so the route silently stops working and nothing on screen says
    why. There were four of these -- File had &Export twice, the Export submenu
    had &C and &S twice each, and View had &c on both "Along c" and "void
    cone".
    """
    from collections import Counter

    from PySide6.QtWidgets import QMenu

    def check(menu, path):
        letters = Counter()
        for action in menu.actions():
            if action.isSeparator():
                continue
            text = action.text()
            if "&" in text:
                index = text.index("&")
                if index + 1 < len(text):
                    letters[text[index + 1].lower()] += 1
            sub = action.menu()
            if isinstance(sub, QMenu):
                check(sub, f"{path} > {text.replace('&', '')}")
        duplicated = {k: v for k, v in letters.items() if v > 1}
        assert not duplicated, f"{path}: {duplicated}"

    for action in window.menuBar().actions():
        if action.menu() is not None:
            check(action.menu(), action.text().replace("&", ""))


def test_every_panel_combo_shows_its_own_items(window, qapp):
    """The anti-elision fix used to be applied to four boxes and forgotten.

    Measured before the sweep: 8 of 17 panel drop-downs were laid out narrower
    than their own longest item and 6 could not show even their current text --
    one had a text field of minus five pixels, its arrow wider than the whole
    widget.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QComboBox, QTabWidget

    window.resize(1739, 940)
    qapp.processEvents()
    tabs = window.findChild(QTabWidget)
    for index in range(tabs.count()):
        tabs.setCurrentIndex(index)
        qapp.processEvents()
        # each page is wrapped in a scroll area, so search the wrapper
        page = tabs.widget(index)
        for box in page.findChildren(QComboBox):
            if not box.isVisibleTo(page) or not box.count():
                continue
            view = box.view()
            assert view is None or view.textElideMode() == Qt.ElideNone, (
                tabs.tabText(index), box.currentText())


def test_the_window_fits_on_a_laptop_screen(window):
    """The whole point of a desktop application is that it opens.

    Measured before the tab pages were allowed to scroll: the minimum size was
    1434 x 1421 px, on a 1739 x 930 screen -- nearly 500 px taller than the
    display, with no way to shrink it. The Volume panel alone demanded 1130 px
    of height and nothing could give way.

    1366 x 768 is the smallest screen worth designing for, and FACET must fit
    inside it with room for the task bar. The width is the harder half, because
    the two docks and the panel column all have real minimums; it is checked
    against a 1440 px screen rather than 1366.
    """
    minimum = window.minimumSizeHint()
    assert minimum.height() <= 730, (
        f"the window cannot be made shorter than {minimum.height()} px, so it "
        f"does not fit a 768 px screen")
    assert minimum.width() <= 1440, (
        f"the window cannot be made narrower than {minimum.width()} px")


def test_all_ten_tabs_are_readable_without_scrolling(window, qapp):
    """The complaint that started this: a tab bar cut off after "Ov...".

    Two separate failures were in play. The names were long enough to need
    636 px of tab bar, and the column holding them was pinned at 400 px with no
    stretch, so it never grew however large the window. Eliding was worse than
    either -- it made every tab unreadable at once rather than hiding two.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QTabWidget

    # The application's own style sheet sets the tab padding, and without it
    # the native one is far looser: the same ten tabs measure 586 px instead of
    # 476. Applying it is what makes this a measurement of FACET.
    chrome.apply(qapp, T.Theme())
    qapp.processEvents()

    tabs = window.findChild(QTabWidget)
    bar = tabs.tabBar()
    assert bar.elideMode() == Qt.ElideNone, (
        "an elided tab bar mangles every name at once")
    # 1739 is this machine's screen; below roughly 1700 the column is narrower
    # than the ten labels need and the bar scrolls, which is a deliberate
    # trade against raising the window's minimum width.
    for width in (1739, 1920):
        window.resize(width, 930)
        qapp.processEvents()
        wanted = sum(bar.tabSizeHint(i).width() for i in range(bar.count()))
        assert wanted <= bar.width(), (
            f"at a {width} px window the ten tabs want {wanted} px of tab bar "
            f"and have {bar.width()}, so the last ones sit behind scroll "
            f"arrows")
