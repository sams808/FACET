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


def test_secondary_text_is_legible_against_its_panel():
    """The hint colour has to carry against the window it is drawn on."""
    def luminance(hexcolor):
        r, g, b = (int(hexcolor[i:i + 2], 16) / 255 for i in (1, 3, 5))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    for factory in T.PRESETS.values():
        c = chrome.ui_colors(factory())
        a, b = sorted((luminance(c.muted), luminance(c.window)))
        contrast = (b + 0.05) / (a + 0.05)
        assert contrast > 3.0, (factory().name, contrast)


def test_accent_text_is_legible_on_the_accent():
    for factory in T.PRESETS.values():
        t = factory()
        c = chrome.ui_colors(t)

        def luminance(hexcolor):
            r, g, b = (int(hexcolor[i:i + 2], 16) / 255 for i in (1, 3, 5))
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        a, b = sorted((luminance(c.accent), luminance(c.accent_text)))
        assert (b + 0.05) / (a + 0.05) > 4.5, t.name


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
                  "ships no bond-valence compilation"):
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
    w.close()


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
