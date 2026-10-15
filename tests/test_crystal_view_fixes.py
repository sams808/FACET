"""Three faults seen in the window on 2026-10-15, each pinned:

* a polyhedron's corners sat a bond length from any oxygen on the symmetry
  copies of a site (the representative's hull was translated, not rotated):
  every corner now coincides with a drawn ligand;
* an isolation carried over to the next structure opened: it belongs to
  the structure it was asked on;
* a click on a bond answered with the atom behind it on the OpenGL tier
  (the pick pass draws atoms only): the stick wins unless the atom's own
  silhouette holds the click.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import dispose

from facet.core import prefs

EXAMPLES = Path(__file__).resolve().parent.parent / "facet" / "data" / "examples"
EULYTITE = EXAMPLES / "eulytite_Bi4SiO4_3_cod9012894.cif"
QUARTZ = EXAMPLES / "quartz_SiO2_cod9013321.cif"


@pytest.fixture
def window(qapp):
    prefs.set("tour_offered", True)
    from facet.ui.preview import PreviewWindow
    w = PreviewWindow()
    yield w
    w.close()
    dispose(w)


def test_every_polyhedron_corner_is_a_drawn_ligand(qapp, window):
    from scipy.spatial import cKDTree
    from facet.ui.preview import PolyhedraMode
    window.load(str(EULYTITE))
    bi = next(r for r in window.results if r.element == "Bi")
    window._on_site_picked(bi.site_index)
    window.poly_mode = PolyhedraMode.SELECTED
    window._rebuild()
    s = window.view.scene
    assert s.n_poly_triangles > 0
    verts = np.asarray(s.poly_vertices, float).reshape(-1, 3)
    d, j = cKDTree(np.asarray(s.atom_position, float)).query(verts)
    assert d.max() < 1e-4
    assert set(np.asarray(s.atom_element)[j]) == {"O"}
    # every copy of the site in the cell: each of its drawn ligands is a corner
    n_copies = int((np.asarray(s.atom_site[:len(window.structure.atoms)]) == bi.site_index).sum())
    assert n_copies == 16
    corner_tree = cKDTree(verts)
    ends = np.asarray(s.bond_atoms, int)
    above = np.asarray(s.bond_valence) >= s.v_bond
    for atom in range(len(window.structure.atoms)):
        if int(s.atom_site[atom]) != bi.site_index:
            continue
        tips = np.vstack([np.asarray(s.bond_b)[above & (ends[:, 0] == atom)],
                          np.asarray(s.bond_a)[above & (ends[:, 1] == atom)]])
        assert len(tips) == bi.cn_valence
        assert corner_tree.query(tips)[0].max() < 1e-4


def test_an_isolation_belongs_to_the_structure_it_was_asked_on(qapp, window):
    window.load_many([str(QUARTZ), str(EULYTITE)])
    window.project.set_active(0)
    window.structure_panel.refresh()
    window._show_entry()
    first = window.view.scene
    si = int(np.flatnonzero(np.asarray(first.atom_element) == "Si")[0])
    window._isolate(si, 1)
    assert window.view.scene.n_atoms == 5
    # switching to the other structure draws it whole
    window.project.set_active(1)
    window.structure_panel.refresh()
    window._show_entry()
    assert window._isolated is None
    assert window.view.scene.n_atoms > 100
    # and back: the first is whole again as well
    window.project.set_active(0)
    window.structure_panel.refresh()
    window._show_entry()
    assert window.view.scene.n_atoms == first.n_atoms


def test_a_click_on_a_stick_wins_over_the_atom_behind_it(qapp, window):
    from PySide6.QtCore import QPoint
    window.load(str(QUARTZ))
    view = window.view
    view.resize(640, 480)
    s = view.scene
    # a bond whose midpoint lies in front; the click lands on the stick
    ends = np.asarray(s.bond_atoms, int)
    above = np.asarray(s.bond_valence) >= s.v_bond
    mid = 0.5 * (np.asarray(s.bond_a) + np.asarray(s.bond_b))
    px = view._visible(mid)
    front = np.flatnonzero(above & (px[:, 2] < 0))
    assert len(front)
    i = int(front[0])
    pos = QPoint(int(round(px[i, 0])), int(round(px[i, 1])))
    assert view.pick_bond_at(pos) == i
    picked = []
    view.bondPicked.connect(picked.append)
    view._handle_select_click(pos)
    # the click is on the stick: the bond is reported, not an atom behind it
    assert picked == [i]
    assert view._selected is None
    # the atom's own silhouette holds a click on the atom, even with a stick behind
    a = int(ends[i, 0])
    pa = view._visible(np.asarray(s.atom_position)[a][None, :])[0]
    picked.clear()
    at_centre = QPoint(int(round(pa[0])), int(round(pa[1])))
    under = view.pick_at(at_centre)
    assert under is not None
    expected_site = int(s.atom_site[under])
    view._handle_select_click(at_centre)
    qapp.processEvents()
    # no bond: the atom's site is selected in the window (the window rebuilds
    # on a pick, so the selection is read through the window)
    assert picked == []
    assert window.site_index == expected_site


def test_the_contacts_below_the_threshold_can_be_left_undrawn(qapp, window):
    # eulytite's Bi: three contacts near 0.4 v.u. and three near 0.12, so a
    # threshold of 0.3 leaves listed contacts below the cut
    window.load(str(EULYTITE))
    window._on_threshold(0.3)
    window._recolour_timer.timeout.emit()
    qapp.processEvents()
    s = window.view.scene
    v = np.asarray(s.bond_valence)
    assert (v < s.v_bond).any() and (v >= s.v_bond).any()
    n_all = s.n_bonds
    assert window.subthreshold_action.isChecked()
    window.subthreshold_action.setChecked(False)
    s = window.view.scene
    assert s.n_bonds == int((v >= window.project.v_bond).sum()) < n_all
    assert (np.asarray(s.bond_valence) >= s.v_bond).all()
    assert len(s.bond_atoms) == s.n_bonds
    assert prefs.get("show_subthreshold") is False
    # a threshold move with the contacts hidden rebuilds once the drag settles
    window._on_threshold(1.5)
    window._recolour_timer.timeout.emit()
    assert window.view.scene.n_bonds == 0
    window.subthreshold_action.setChecked(True)
    assert window.view.scene.n_bonds == n_all
