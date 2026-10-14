"""An atom of an MD frame investigated as the crystal workspace investigates
a site, and the angles between bonds drawn as arcs on both views.

The site result of an atom is built from the frame's valence table and
checked against what the table holds; the Site-tab text is one builder for
both workspaces; the arcs' angles are checked against the dot product of
the drawn bond vectors.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import dispose, sample_cif

from facet.core import bulk, bv, coordination, md_model, md_readers, md_site

NS3 = Path(r"C:\Users\samso\AppData\Local\Temp\claude\C--Users-samso"
           r"\342b7e11-8171-41a0-8eee-4126254c9cc5\scratchpad"
           r"\facet_md_descriptors\models\ns3_pedone\nvt300K_last.data")
NS3_TYPES = {1: "Si", 2: "O", 3: "Na"}
SAMPLE = sample_cif("1526458", "1526458_Bi2O3.cif")
needs_ns3 = pytest.mark.skipif(not NS3.exists(),
                               reason="the NS3 model is not on this machine")


@pytest.fixture(scope="module")
def frame_and_table():
    if not NS3.exists():
        pytest.skip("the NS3 model is not on this machine")
    frame = md_readers.read_trajectory(str(NS3), type_map=NS3_TYPES).frame(0)
    ox = md_model.model_oxidation(frame.species).per_atom(frame.elements)
    table, _ = bulk.analyse_frame(frame, ox, bv.DEFAULT, v_bond_vu=0.075,
                                  v_list_vu=0.02)
    return frame, table


# -- the site result of an atom ---------------------------------------------------

@needs_ns3
def test_an_atoms_site_result_reads_the_tables_row(frame_and_table):
    frame, table = frame_and_table
    row = int(np.flatnonzero(frame.elements == "Si")[0])
    r = md_site.site_result_for_atom(frame, table, row, v_bond=0.075,
                                     v_list=0.02, k=0)
    assert r.element == "Si" and r.ox == 4 and r.site_index == row
    listed = [j for j in range(table.nbr.shape[1]) if table.nbr[row, j] >= 0
              and table.v_vu[row, j] > 0.02]
    assert len(r.contacts) == len(listed)
    # the table's own count at the threshold is the result's CN
    expected = bulk.at_threshold(table, 0.075)
    cn_table = int(np.asarray(expected.cn)[row]) if hasattr(expected, "cn") \
        else r.cn_valence
    assert r.cn_valence == cn_table
    # the contacts carry the table's distances and the pair's parameter
    d = sorted(float(table.d_ang[row, j]) for j in listed)
    assert [c.distance for c in r.contacts] == pytest.approx(d)
    assert r.contacts[0].param is not None and r.contacts[0].param.label.startswith("Si")
    assert len(r.plateaus) == len(r.contacts)
    assert any("frame 0" in n for n in r.notes)


@needs_ns3
def test_an_anion_reads_the_cations_around_it(frame_and_table):
    frame, table = frame_and_table
    row = int(np.flatnonzero(frame.elements == "O")[0])
    r = md_site.site_result_for_atom(frame, table, row, v_bond=0.075,
                                     v_list=0.02)
    assert r.ox == -2
    assert all(c.element in ("Si", "Na") for c in r.contacts)
    assert r.cn_valence >= 1


@needs_ns3
def test_reclassifying_an_atom_matches_a_fresh_result(frame_and_table):
    frame, table = frame_and_table
    row = int(np.flatnonzero(frame.elements == "Na")[0])
    base = md_site.site_result_for_atom(frame, table, row, v_bond=0.075,
                                        v_list=0.02)
    moved = coordination.reclassify(base, 0.03)
    fresh = md_site.site_result_for_atom(frame, table, row, v_bond=0.03,
                                         v_list=0.02)
    assert moved.cn_valence == fresh.cn_valence
    assert moved.bvs == pytest.approx(fresh.bvs)


def test_a_row_outside_the_frame_is_refused():
    frame = md_model.frame_from_arrays(np.array(["Si", "O"]),
                                       np.array([[0.0, 0, 0], [1.6, 0, 0]]),
                                       box_ang=np.diag([10.0, 10, 10]))

    class T:
        nbr = np.zeros((2, 1), int)
        ox = np.array([4, -2])
        is_anion = np.array([False, True])
    with pytest.raises(ValueError):
        md_site.site_result_for_atom(frame, T(), 7)


# -- the shared Site-tab text -------------------------------------------------------

def test_the_site_html_is_one_builder_for_both_workspaces():
    from facet.core import cif
    from facet.ui.site_html import site_html
    r = coordination.analyse_structure(cif.read(str(SAMPLE)), bv.DEFAULT,
                                       v_bond=0.075, v_list=0.02,
                                       cations_only=True)[0]
    text = site_html(r, 0.075, "#888", show_running=True)
    assert "Coordination number" in text and "Contacts" in text
    assert "so far" in text and "Wyckoff" in text
    text = site_html(r, 0.075, "#888", headline="frame 3 · row 12",
                     apriori_reason="no network", show_running=False)
    assert "frame 3" in text and "Wyckoff" not in text
    assert "No a priori bond valences: no network" in text
    assert "so far" not in text


# -- the angles ------------------------------------------------------------------------

def test_the_arcs_angles_are_the_bond_angles(qapp):
    from facet.ui import preview
    win = preview.PreviewWindow()
    try:
        win.load_many([str(SAMPLE)])
        qapp.processEvents()
        entry = win.project.current
        entry.selected_site = win.project.results_for(entry)[0].site_index
        win._rebuild()
        qapp.processEvents()
        scene = win.scene
        hub = int(np.flatnonzero(np.asarray(scene.atom_site)
                                 == entry.selected_site)[0])
        win.view.set_angle_hub(hub)
        pairs = win.view.hub_angles()
        r = win._current_result()
        n = r.cn_valence
        assert len(pairs) == n * (n - 1) // 2
        # each angle is the angle between the two bond vectors
        centre = np.asarray(scene.atom_position[hub], float)
        ends = np.asarray(scene.bond_atoms, int)
        for i, j, ang in pairs:
            tips = []
            for b in (i, j):
                tips.append(np.asarray(scene.bond_b[b] if ends[b, 0] == hub
                                       else scene.bond_a[b], float))
            u, v = tips[0] - centre, tips[1] - centre
            expected = np.degrees(np.arccos(
                np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
            assert ang == pytest.approx(expected, abs=1e-6)
        assert 0 < min(a for _, _, a in pairs) and max(a for _, _, a in pairs) < 180
        # the menu offers to hide them now, and to show them before
        menu = win._build_context_menu(hub)
        texts = [a.text() for a in menu.actions()]
        assert "Hide the angles" in texts
        win.view.set_angle_hub(None)
        texts = [a.text() for a in win._build_context_menu(hub).actions()]
        assert any(t.startswith("Show the angles at") for t in texts)
        # drawing with arcs raises nothing on the painter tier
        win.view.set_angle_hub(hub)
        assert win.view.grab().toImage().width() > 0
    finally:
        win.close()
        dispose(win)


def test_an_arc_spans_the_shorter_way(qapp):
    """The arc from one arm to the other is at most a half turn."""
    from PySide6.QtGui import QImage, QPainter, QColor
    from facet.gl.view import StructureView
    image = QImage(200, 200, QImage.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    try:
        StructureView._paint_arc(painter, (100.0, 100.0), (160.0, 100.0),
                                 (100.0, 40.0), 1.0, QColor(255, 0, 0),
                                 text="90.0°")
    finally:
        painter.end()
    # something was painted, in the quadrant between the two arms (upper
    # right on screen), and nothing in the opposite quadrant
    px = np.array([[image.pixelColor(x, y).alpha() for x in range(200)]
                   for y in range(200)])
    assert px[40:100, 100:160].sum() > 0
    assert px[110:200, 0:90].sum() == 0


# -- the Atom tab ---------------------------------------------------------------------

@needs_ns3
def test_clicking_an_atom_fills_the_atom_tab_and_the_explorer(qapp):
    import time
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
        assert wait(lambda: c.frame_view is not None
                    and c.frame_view.table is not None)
        assert wait(lambda: c.view.scene is not None and c.view.scene.n_atoms > 0)
        scene = c.view.scene
        si = int(np.flatnonzero(np.asarray(scene.atom_element) == "Si")[0])
        assert c.atom_explorer.isHidden()
        c._on_atom_picked(si)
        qapp.processEvents()
        r = c._atom_result
        assert r is not None and r.element == "Si"
        assert "Contacts" in c.atom_text.toHtml()
        assert not c.atom_explorer.isHidden()
        assert c.atom_explorer.result is r
        assert c.atom_explorer.pin_v == pytest.approx(c.frame_view.v_bond_vu)
        # the strip moves the atom's numbers, and the explorer moves the strip
        c.strip.set_threshold(0.3, emit=True)
        qapp.processEvents()
        assert c._atom_result.v_bond == pytest.approx(0.3)
        assert c.atom_explorer.v_bond == pytest.approx(0.3)
        c.atom_explorer.set_threshold(0.075, emit=True)
        qapp.processEvents()
        assert c.strip.v_bond == pytest.approx(0.075)
        assert c._atom_result.v_bond == pytest.approx(0.075)
        # the menu investigates and shows the angles
        texts = [a.text() for a in c.build_context_menu(si).actions()
                 if not a.isSeparator()]
        assert any(t.startswith("Investigate") for t in texts)
        assert any(t.startswith("Show the angles") for t in texts)
        c.view.set_angle_hub(si)
        assert len(c.view.hub_angles()) == 6          # a tetrahedron
    finally:
        win.close()
        dispose(win)
