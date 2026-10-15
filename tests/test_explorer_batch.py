"""The explorer and Site-tab batch: a plateau table, a second site overlaid,
the distance axis read against a chosen pair, threshold bookmarks, ECoN on
the CN axis, per-contact sensitivity, the uncertainty band, the angles at
the site, the Site tab as a page, and the keyboard on the explorer.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import dispose, sample_cif

from facet.core import bv, cif, coordination, prefs
from facet.ui.cutoff_explorer import AXIS_DISTANCE, AXIS_VALENCE, CutoffExplorer

SAMPLE = sample_cif("1526458", "1526458_Bi2O3.cif")


@pytest.fixture(scope="module")
def sites():
    return coordination.analyse_structure(cif.read(str(SAMPLE)), bv.DEFAULT,
                                          v_bond=0.075, v_list=0.02,
                                          cations_only=True)


@pytest.fixture
def explorer(qapp, sites):
    w = CutoffExplorer()
    w.resize(1000, 330)
    w.set_result(sites[0])
    w.set_pin(0.075)
    w.set_threshold(0.075)
    yield w
    dispose(w)


def test_the_site_tab_lists_every_plateau_and_the_angles(sites):
    from facet.ui.site_html import site_html
    text = site_html(sites[0], 0.075, "#888")
    assert "Plateaus" in text
    for q in sites[0].plateaus:
        assert f"{q.v_low:.4f}&ndash;{q.v_high:.4f}" in text
    assert text.count("&#9679;</td></tr>") >= 1       # the current tread marked
    assert "Bond angles at the site" in text and "spread" in text


def test_a_second_site_overlays_and_widens_the_axis(explorer, sites):
    lo0, hi0 = explorer._v_range()
    explorer.set_second(sites[1])
    assert explorer.second is sites[1]
    lo, hi = explorer._v_range()
    assert lo <= lo0 and hi >= hi0
    assert explorer.grab().toImage().width() > 0
    explorer.set_second(None)
    assert explorer.second is None


def test_the_axis_can_be_read_against_a_named_pair(explorer):
    pairs = explorer.pairs()
    assert pairs and pairs[0].startswith("Bi")
    explorer.set_axis_pair(pairs[0])
    assert explorer._dominant_param().label == pairs[0]
    explorer.set_axis_pair("Nothing-X")
    assert explorer._dominant_param().label == pairs[0]      # falls back


def test_bookmarks_are_marks_the_cursor_jumps_to(explorer):
    explorer.add_bookmark("paper", 0.12)
    explorer.add_bookmark("here")
    assert explorer.bookmarks["here"] == pytest.approx(0.075)
    labels = [label for _, label, _ in explorer._reference_marks()]
    assert any("paper" in l for l in labels)
    seen = []
    explorer.thresholdChanged.connect(seen.append)
    explorer.jump_to(0.12)
    assert seen and seen[-1] == pytest.approx(0.12)
    explorer.remove_bookmark("paper")
    assert "paper" not in explorer.bookmarks


def test_sensitivity_says_what_crossing_the_cut_does(explorer, sites):
    r = sites[0]
    inside = next(i for i, c in enumerate(r.contacts)
                  if c.has_valence and c.valence > 0.075)
    outside = next(i for i, c in enumerate(r.contacts)
                   if c.has_valence and c.valence <= 0.075)
    d_in, phi_in = explorer.sensitivity(inside)
    d_out, phi_out = explorer.sensitivity(outside)
    assert d_in == pytest.approx(-r.contacts[inside].valence)
    assert d_out == pytest.approx(r.contacts[outside].valence)
    assert phi_in == phi_in and phi_out == phi_out


def test_the_band_and_econ_draw_and_can_be_turned_off(explorer):
    for name in ("show_band", "show_econ"):
        assert getattr(explorer, name)
        explorer.set_option(name, False)
        assert explorer.grab().toImage().width() > 0
        explorer.set_option(name, True)
    with pytest.raises(ValueError):
        explorer.set_option("show_moon", True)


def test_the_keyboard_moves_the_threshold(explorer, qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    explorer.show()
    explorer.setFocus()
    v0 = explorer.v_bond
    QTest.keyClick(explorer, Qt.Key_Right)
    assert explorer.v_bond == pytest.approx(v0 * 1.02)
    QTest.keyClick(explorer, Qt.Key_Left)
    assert explorer.v_bond == pytest.approx(v0)
    QTest.keyClick(explorer, Qt.Key_PageUp)
    up = explorer.v_bond
    assert up > v0
    plateau = next(q for q in explorer.result.plateaus if q.contains(up))
    assert plateau.v_low < up < plateau.v_high
    QTest.keyClick(explorer, Qt.Key_Home)
    assert explorer.v_bond == pytest.approx(0.075)


def test_the_window_offers_overlay_bookmarks_pairs_and_the_site_page(qapp, tmp_path):
    prefs.set("tour_offered", True)
    from facet.ui import preview
    win = preview.PreviewWindow()
    try:
        win.load_many([str(SAMPLE)])
        qapp.processEvents()
        entry = win.project.current
        results = win.project.results_for(entry)
        entry.selected_site = results[0].site_index
        win._rebuild()
        qapp.processEvents()
        menu = win._build_site_menu(1)
        texts = [a.text() for a in menu.actions() if not a.isSeparator()]
        assert "Overlay on the explorer" in texts
        win.overlay_site(results[1].site_index)
        assert win.explorer.second is not None
        assert win.explorer.second.site_index == results[1].site_index
        win.add_bookmark("convention", 0.1)
        assert prefs.get("bookmarks") == {"convention": 0.1}
        win._fill_bookmark_menu()
        texts = [a.text().replace("&", "") for a in win.bookmark_menu.actions()
                 if not a.isSeparator()]
        assert any(t.startswith("convention") for t in texts)
        win._fill_axis_pair_menu()
        texts = [a.text() for a in win.axis_pair_menu.actions()]
        assert texts and texts[0].startswith("Bi")
        out = win.export_site_html(tmp_path / "site.html")
        page = out.read_text(encoding="utf-8")
        assert "Coordination number" in page and "data:image/png;base64," in page
        assert "Plateaus" in page
    finally:
        win.close()
        dispose(win)
