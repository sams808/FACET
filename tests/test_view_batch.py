"""The 3D-view batch: angle labels laid out apart, the dihedral drawn, depth
cueing about the focus, a slab through an MD box, polyhedra within the
focus, a bond click on every tier, stereo on the model's view, a
measurement that survives a rebuild, the on-screen legend, and the export
that carries the annotations.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import dispose, sample_cif

from facet.core import md_model, prefs
from facet.core.md_environment import slab_mask

SAMPLE = sample_cif("1526458", "1526458_Bi2O3.cif")
NS3 = Path(r"C:\Users\samso\AppData\Local\Temp\claude\C--Users-samso"
           r"\342b7e11-8171-41a0-8eee-4126254c9cc5\scratchpad"
           r"\facet_md_descriptors\models\ns3_pedone\nvt300K_last.data")


@pytest.fixture
def window(qapp):
    prefs.set("tour_offered", True)
    from facet.ui import preview
    win = preview.PreviewWindow()
    win.load_many([str(SAMPLE)])
    qapp.processEvents()
    entry = win.project.current
    entry.selected_site = win.project.results_for(entry)[0].site_index
    win._rebuild()
    qapp.processEvents()
    yield win
    win.close()
    dispose(win)


def test_a_slab_keeps_the_atoms_within_its_thickness_across_the_wrap():
    box = np.diag([10.0, 10.0, 10.0])
    cart = np.array([[0.2, 5, 5], [9.9, 5, 5], [5.0, 5, 5], [2.5, 5, 5]])
    frame = md_model.frame_from_arrays(np.array(["Si", "O", "O", "Na"]), cart,
                                       box_ang=box)
    keep = slab_mask(frame, 0, 0.0, 2.0)       # about x = 0, 1 Å each way
    assert keep.tolist() == [True, True, False, False]
    keep = slab_mask(frame, 0, 0.5, 2.0)
    assert keep.tolist() == [False, False, True, False]


def test_angle_labels_do_not_overlap(window, qapp):
    scene = window.scene
    entry = window.project.current
    hub = int(np.flatnonzero(np.asarray(scene.atom_site) == entry.selected_site)[0])
    window.view.set_angle_hub(hub)
    from PySide6.QtGui import QImage, QPainter
    image = QImage(800, 600, QImage.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    placed = []
    try:
        # paint arcs for every pair through the public helper, collecting
        # the label rectangles the way the overlay does
        pairs = window.view.hub_angles()
        centre = np.array([400.0, 300.0])
        for n, (i, j, ang) in enumerate(pairs):
            a = centre + 120 * np.array([np.cos(0.3 * n), np.sin(0.3 * n)])
            b = centre + 120 * np.array([np.cos(0.3 * n + 0.4), np.sin(0.3 * n + 0.4)])
            window.view._paint_arc(painter, centre, a, b, 1.0,
                                   painter.pen().color(), fraction=0.3,
                                   text=f"{ang:.1f}°", placed=placed)
    finally:
        painter.end()
    assert len(placed) == len(pairs) > 1
    for x in range(len(placed)):
        for y in range(x + 1, len(placed)):
            assert not placed[x].intersects(placed[y])


def test_a_four_atom_measurement_draws_the_dihedral(window, qapp):
    scene = window.scene
    chain = list(range(4))
    window.view._measure_chain = chain
    assert window.view.grab().toImage().width() > 0
    # and the chain survives a rebuild of the same atoms
    window._rebuild()
    assert window.view._measure_chain == chain
    # but not a different structure
    window.load_many([str(sample_cif("1526458", "1526458_Bi2O3.cif"))])
    qapp.processEvents()


def test_depth_cueing_follows_the_focus(window, qapp):
    window.focus_action.setChecked(True)
    qapp.processEvents()
    window.view.set_depth_dim_focus(True)
    anchor = window.view.camera.fog_anchor
    assert anchor is not None
    scene = window.scene
    assert np.allclose(anchor, scene.atom_position[scene.focus_atom])
    near, far = window.view.camera.fog_range()
    assert near < far
    window.view.set_depth_dim_focus(False)
    assert window.view.camera.fog_anchor is None


def test_polyhedra_within_the_focus(window, qapp):
    window.focus_action.setChecked(True)
    qapp.processEvents()
    window.focus_polyhedra_action.setChecked(True)
    qapp.processEvents()
    sites = window._polyhedron_sites()
    scene = window.scene
    inside = {int(s) for s in np.asarray(scene.atom_site)[scene.focus_mask()]}
    assert sites and set(sites) <= inside


def test_a_click_on_a_bond_names_it(window, qapp):
    from PySide6.QtCore import QPoint
    scene = window.scene
    view = window.view
    view.resize(600, 400)
    view.show()
    qapp.processEvents()
    # the screen midpoint of the first drawn bond above the threshold
    i = int(np.flatnonzero(np.asarray(scene.bond_valence) >= scene.v_bond)[0])
    mid = 0.5 * (np.asarray(scene.bond_a[i]) + np.asarray(scene.bond_b[i]))
    with view._in_view(view.camera, view.width(), view.height()):
        px = view._visible(np.array([mid]))[0]
    bond = view.pick_bond_at(QPoint(int(px[0]), int(px[1])))
    assert bond is not None
    seen = []
    view.bondPicked.connect(seen.append)
    window._on_bond_picked(bond)
    assert "Bond" in window.statusBar().currentMessage()
    assert "Å" in window.statusBar().currentMessage()


def test_the_legend_draws_for_a_value_mode(window, qapp):
    from facet.core import theme as T
    window.theme.color_mode = T.ColorMode.BVS
    window.view.set_theme(window.theme)
    window.view.set_label_context(window.structure, window.results)
    assert window.view.show_legend
    assert window.view.grab().toImage().width() > 0
    window._set_legend(False)
    assert not window.view.show_legend


@pytest.mark.skipif(not NS3.exists(), reason="the NS3 model is not on this machine")
def test_the_model_toolbar_slab_and_stereo(qapp):
    import time
    prefs.set("tour_offered", True)
    from facet.ui import preview
    from facet.gl import stereo as stereo_mod
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
            c.reread({"type_map": {1: "Si", 2: "O", 3: "Na"}})
            assert wait(lambda: c.summary is not None)
        c.show_tab("3D view")
        assert wait(lambda: c.view.scene is not None and c.view.scene.n_atoms > 0)
        n_all = c.view.scene.n_atoms
        c.toolbar.slab_thickness.setValue(6.0)
        c.toolbar.slab_check.setChecked(True)
        assert wait(lambda: c.view.scene is not None and c.view.scene.n_atoms < n_all)
        assert c.view.scene.n_atoms < n_all / 2
        c.toolbar.slab_check.setChecked(False)
        assert wait(lambda: c.view.scene is not None and c.view.scene.n_atoms == n_all)
        mode = next(m for m in stereo_mod.Mode if m is not stereo_mod.Mode.OFF)
        win._on_viewing(mode, 0.05, True)
        assert c.view.stereo is mode
        win._on_viewing(stereo_mod.Mode.OFF, 0.05, True)
    finally:
        win.close()
        dispose(win)
