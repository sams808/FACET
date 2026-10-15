"""The highlight-and-selection batch: rules by angle, by neighbour
composition, by ring membership and by distance to a channel; saved rule
sets; a rule from a dragged range on a figure; a selection gathered with
Ctrl+click; counts per element; colour by shell from the atom; and the
highlighted set on the Results figure.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import dispose

from facet.core import bulk, bv, md_model, md_readers, prefs
from facet.ui.md_highlight import FrameData, Rule, _keep_of_rule, apply_rules

NS3 = Path(r"C:\Users\samso\AppData\Local\Temp\claude\C--Users-samso"
           r"\342b7e11-8171-41a0-8eee-4126254c9cc5\scratchpad"
           r"\facet_md_descriptors\models\ns3_pedone\nvt300K_last.data")
NS3_TYPES = {1: "Si", 2: "O", 3: "Na"}
needs_ns3 = pytest.mark.skipif(not NS3.exists(),
                               reason="the NS3 model is not on this machine")


@pytest.fixture(scope="module")
def fd():
    if not NS3.exists():
        pytest.skip("the NS3 model is not on this machine")
    frame = md_readers.read_trajectory(str(NS3), type_map=NS3_TYPES).frame(0)
    ox = md_model.model_oxidation(frame.species).per_atom(frame.elements)
    table, _ = bulk.analyse_frame(frame, ox, bv.DEFAULT, v_bond_vu=0.075,
                                  v_list_vu=0.02)
    return FrameData.from_frame(frame, table, k=0, v_bond=0.075,
                                formers=("Si",), ox_atom=ox)


# -- the new descriptors ----------------------------------------------------------

@needs_ns3
def test_angles_at_silicon_are_near_tetrahedral(fd):
    mean, dev = fd.angle_stats()
    si = fd.elements == "Si"
    assert np.nanmean(mean[si]) == pytest.approx(109.5, abs=3.0)
    assert np.nanmean(dev[si]) < 10.0
    # an atom with one bond has no angle
    one = fd.cn == 1
    if one.any():
        assert np.isnan(mean[one]).all()
    keep = _keep_of_rule(fd, Rule(kind="show-only", element="Si",
                                  descriptor="tetrahedral deviation",
                                  sense="≤", value=5.0))
    assert 0 < keep.sum() < si.sum()


@needs_ns3
def test_a_neighbour_rule_counts_bonds_to_one_element(fd):
    o = fd.elements == "O"
    to_si = fd.neighbour_count("Si")
    # a bridging oxygen has two Si; a non-bridging one has one
    rule = Rule(kind="neighbours", element="O", neighbour="Si", sense="=",
                value=1)
    nbo = _keep_of_rule(fd, rule)
    assert nbo.sum() > 0 and (to_si[nbo] == 1).all()
    assert (fd.former_cn[nbo] == 1).all()
    assert fd.neighbour_count("any")[o].sum() == pytest.approx(
        (fd.cn[o]).sum())


@needs_ns3
def test_ring_and_channel_descriptors_say_what_they_need(fd):
    with pytest.raises(ValueError, match="rings"):
        fd.descriptor("smallest ring")
    with pytest.raises(ValueError, match="channels"):
        fd.descriptor("distance to channel")
    fd.ring_size = np.full(fd.n_atoms, 6.0)
    assert fd.descriptor("smallest ring")[0] == 6.0
    fd.ring_size = None


@needs_ns3
def test_the_count_line_reads_by_element_and_the_keep_is_returned(fd):
    from facet.gl.scene import Scene
    from facet.ui.md_scene import build_md_scene
    scene = build_md_scene(fd.frame, fd.table, v_bond=0.075)
    out = apply_rules(scene, fd, [Rule(kind="show-only", element="O",
                                       descriptor="CN", sense="≥", value=2,
                                       enabled=True)])
    assert "matched:" in out.note and " O" in out.note
    assert out.keep is not None and out.keep.sum() == out.n_matching


@needs_ns3
def test_colour_by_shell_from_the_atom(fd):
    from facet.ui.md_highlight import colour_atoms
    from facet.ui.md_scene import build_md_scene
    scene = build_md_scene(fd.frame, fd.table, v_bond=0.075)
    note = colour_atoms(scene, fd, "shell from the atom")
    assert "needs an atom" in note
    fd.shell_centre = int(np.flatnonzero(fd.elements == "Si")[0])
    note = colour_atoms(scene, fd, "shell from the atom")
    assert note == ""
    shell = fd.shells_from(fd.shell_centre, 3)
    assert shell[fd.shell_centre] == 0 and (shell == 1).sum() == fd.cn[fd.shell_centre]
    fd.shell_centre = None


# -- rule sets --------------------------------------------------------------------

def test_rule_sets_round_trip_through_the_preferences(qapp):
    from facet.ui.md_highlight import HighlightPage
    page = HighlightPage()
    try:
        for r in list(page.rows):
            page.remove_rule(r)
        row = page.add_rule("neighbours")
        row.sense.setCurrentIndex(row.sense.findText("="))
        row.value.setValue(2)
        row.tick.setChecked(True)
        page.save_set("bridging")
        assert "bridging" in (prefs.get("rule_sets") or {})
        for r in list(page.rows):
            page.remove_rule(r)
        assert page.rules() == []
        assert page.load_set("bridging")
        rules = page.rules()
        assert len(rules) == 1 and rules[0].kind == "neighbours" \
            and rules[0].sense == "=" and rules[0].value == 2 and rules[0].enabled
        assert not page.load_set("nothing")
    finally:
        dispose(page)


# -- a rule from a range on a figure ----------------------------------------------

def test_a_dragged_range_becomes_two_rules(qapp):
    from facet.ui.md_views import ResultBrowser
    browser = ResultBrowser()
    try:
        seen = []
        browser.highlightRequested.connect(lambda rules, words: seen.append((rules, words)))
        browser._current = ("item", "glass", "BVS Si (BV)")

        class R:
            outputs = {}
        browser.result = R()
        browser._on_range_selected(3.8, 4.2)
        assert seen and [r["sense"] for r in seen[0][0]] == ["≥", "≤"]
        assert seen[0][0][0]["value"] == 3.8 and seen[0][0][1]["value"] == 4.2
        assert "Si" in seen[0][1]
    finally:
        dispose(browser)


def test_the_plot_emits_a_range_on_control_drag(qapp):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from facet.ui.md_plot import StatPlot
    plot = StatPlot()
    plot.resize(500, 300)
    plot.show()
    qapp.processEvents()
    seen = []
    plot.rangeSelected.connect(lambda lo, hi: seen.append((lo, hi)))
    QTest.mousePress(plot, Qt.LeftButton, Qt.ControlModifier, QPoint(100, 150))
    QTest.mouseRelease(plot, Qt.LeftButton, Qt.ControlModifier, QPoint(300, 150))
    assert len(seen) == 1 and seen[0][0] < seen[0][1]
    dispose(plot)


# -- the highlighted set on the Results figure ----------------------------------

@needs_ns3
def test_the_highlighted_atoms_become_a_series(fd):
    from facet.core.md_stats import Distribution
    from facet.ui.md_views import _subset_series
    keep = (fd.elements == "Si") & (fd.qn == 4)
    d = Distribution(name="Qn Si (BV)", keys=(2, 3, 4), kind="fraction",
                     per_frame=np.array([[0.1, 0.3, 0.6]]), n_items=np.array([750]))
    series = _subset_series((fd, keep), "Qn Si (BV)", d)
    assert series is not None and series.label.startswith("highlighted")
    assert series.mean[2] == pytest.approx(1.0) and series.mean[:2].sum() == 0
    assert _subset_series(None, "Qn Si (BV)", d) is None
    assert _subset_series((fd, keep), "rings", d) is None


# -- the selection on the view ------------------------------------------------------

@needs_ns3
def test_control_click_gathers_a_selection_and_the_menu_acts_on_it(qapp, tmp_path):
    import time
    prefs.set("tour_offered", True)
    from facet.ui import preview
    win = preview.PreviewWindow()
    try:
        win.load_many([str(NS3)])
        qapp.processEvents()
        c = win.project.entries[0].controller

        def wait(cond, s=90):
            end = time.monotonic() + s
            while time.monotonic() < end and not cond():
                qapp.processEvents()
                time.sleep(0.05)
            return cond()

        if c.summary is None:
            c.reread({"type_map": NS3_TYPES})
            assert wait(lambda: c.summary is not None)
        c.show_tab("3D view")
        assert wait(lambda: c.view.scene is not None and c.view.scene.n_atoms > 0)
        scene = c.view.scene
        si = [int(i) for i in np.flatnonzero(np.asarray(scene.atom_element) == "Si")[:3]]
        seen = []
        c.view.selectionChanged.connect(seen.append)
        c.view.set_selection(si)
        assert c.view.selection == si and seen[-1] == si
        rows = c.selected_rows()
        assert len(rows) == 3
        texts = [a.text() for a in c.build_context_menu(si[0]).actions()
                 if not a.isSeparator()]
        assert any(t.startswith("Isolate the 3 selected") for t in texts)
        assert any(t.startswith("Export the 3 selected") for t in texts)
        out = c.export_selection(tmp_path / "sel.xyz")
        assert out.read_text(encoding="utf-8").splitlines()[0] == "3"
        n_all = scene.n_atoms
        c.isolate(rows, 1)
        assert wait(lambda: c.view.scene is not None and c.view.scene.n_atoms < n_all)
        assert c.view.scene.n_atoms >= 3 + 4
        c.isolate(None)
        assert wait(lambda: c.view.scene is not None and c.view.scene.n_atoms == n_all)
        c.view.set_selection([])
        assert c.view.selection == []
    finally:
        win.close()
        dispose(win)
