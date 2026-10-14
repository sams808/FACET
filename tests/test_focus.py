"""The focus: dim everything but the selected site and what lies within the
distance the bond threshold corresponds to, and draw that sphere.

The sphere's radius is the explorer's distance axis read at the cursor, so
dragging the cursor grows and shrinks it. The atoms it keeps are the ones
within that distance of the focused atom -- a geometric fact, tested here
against the scene's own positions -- and clearing the focus restores every
colour and radius exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import dispose, sample_cif

from facet.gl import vectors

SAMPLE = sample_cif("1526458", "1526458_Bi2O3.cif")


# -- the sphere ------------------------------------------------------------------

def test_the_sphere_mesh_lies_on_its_radius_with_outward_normals():
    mesh = vectors.sphere_mesh((1.0, 2.0, 3.0), 2.5, (1, 0.5, 0), alpha=0.2)
    centre = np.array([1.0, 2.0, 3.0])
    r = np.linalg.norm(mesh.vertices - centre, axis=1)
    assert np.allclose(r, 2.5, atol=1e-5)
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-5)
    outward = ((mesh.vertices - centre) * mesh.normals).sum(axis=1)
    assert (outward > 0).all()
    assert len(mesh.vertices) % 3 == 0
    assert len(mesh.part_of) == len(mesh.vertices) // 3
    assert mesh.alpha == pytest.approx(0.2)


# -- the scene -------------------------------------------------------------------

@pytest.fixture
def window(qapp):
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


def test_focus_keeps_exactly_the_atoms_within_the_radius(window, qapp):
    scene = window.scene
    before_c = np.array(scene.atom_color)
    before_r = np.array(scene.atom_radius)
    window.focus_action.setChecked(True)
    qapp.processEvents()
    scene = window.scene
    assert scene.focus_atom is not None
    centre = scene.atom_position[scene.focus_atom]
    d = np.linalg.norm(scene.atom_position - centre[None, :], axis=1)
    inside = d <= scene.focus_radius + 1e-6
    assert np.array_equal(scene.focus_mask(), inside)
    # the kept atoms keep their colour and size; the rest are paler and smaller
    assert np.allclose(scene.atom_color[inside], before_c[inside])
    assert np.allclose(scene.atom_radius[inside], before_r[inside])
    assert (scene.atom_radius[~inside] < before_r[~inside]).all()
    assert not np.allclose(scene.atom_color[~inside], before_c[~inside])
    # the sphere is drawn, at the focused atom, with the radius of the cut
    spheres = [m for m in scene.overlay_meshes
               if len(m.anchors) == 1 and np.allclose(m.anchors[0], centre)]
    assert spheres and 0 < spheres[0].alpha < 1
    assert np.allclose(np.linalg.norm(spheres[0].vertices - centre, axis=1),
                       scene.focus_radius, atol=1e-4)


def test_the_radius_is_the_distance_the_threshold_means(window, qapp):
    window.focus_action.setChecked(True)
    qapp.processEvents()
    result = window._current_result()
    param = next(c.param for c in result.contacts if c.param is not None)
    for v in (0.03, 0.2):
        window.explorer.set_threshold(v, emit=True)
        qapp.processEvents()
        assert window.scene.focus_radius == pytest.approx(
            float(param.distance_for(v)))
    # a lower threshold reaches further, so it keeps at least as many atoms
    window.explorer.set_threshold(0.2, emit=True)
    qapp.processEvents()
    tight = int(window.scene.focus_mask().sum())
    window.explorer.set_threshold(0.03, emit=True)
    qapp.processEvents()
    assert int(window.scene.focus_mask().sum()) >= tight
    # and the number of ligands inside is the coordination number at the cut
    r = window._current_result()
    assert int(window.scene.focus_mask().sum()) - 1 >= r.cn_valence


def test_clearing_the_focus_restores_the_scene(window, qapp):
    scene = window.scene
    colours = np.array(scene.atom_color)
    radii = np.array(scene.atom_radius)
    bonds = np.array(scene.bond_radius)
    window.focus_action.setChecked(True)
    qapp.processEvents()
    window.focus_action.setChecked(False)
    qapp.processEvents()
    scene = window.scene
    assert scene.focus_atom is None
    assert np.allclose(scene.atom_color, colours)
    assert np.allclose(scene.atom_radius, radii)
    assert np.allclose(scene.bond_radius, bonds)
    assert not any(len(m.anchors) == 1 and m.alpha < 1 for m in scene.overlay_meshes)


def test_the_focus_survives_a_rebuild_and_follows_the_site(window, qapp):
    window.focus_action.setChecked(True)
    qapp.processEvents()
    first = window.scene.focus_atom
    entry = window.project.current
    other = window.project.results_for(entry)[1].site_index
    entry.selected_site = other
    window._rebuild()
    qapp.processEvents()
    scene = window.scene
    assert scene.focus_atom is not None
    assert int(scene.atom_site[scene.focus_atom]) == other
    assert scene.focus_atom != first or scene.atom_site[first] == other


def test_a_scene_without_a_selected_site_has_no_focus(qapp):
    from facet.ui import preview
    win = preview.PreviewWindow()
    try:
        win.load_many([str(SAMPLE)])
        qapp.processEvents()
        win.project.current.selected_site = None
        win._rebuild()
        win.focus_action.setChecked(True)
        qapp.processEvents()
        assert win.scene.focus_atom is None
    finally:
        win.close()
        dispose(win)
