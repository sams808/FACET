"""The basicity field: the basicity a probe cation would sense at every
point, and the gradient, slice and isolation it is drawn with.

What is pinned:

* at an anion's own position the field reads that anion's lambda, and
  with one anion site the field is flat (it interpolates anion values; it
  is not a density);
* the probe's valence grid is the existing bond-valence map, term for term,
  so the refactor that made both sums share one pass changed nothing;
* an MD frame that is the crystal tiled gives the same field at the same
  points, through the other code path (a KD-tree over image points);
* below the valence floor the field is undefined, and a window or a region
  leaves undefined exactly the points outside it;
* the gradient is one translucent mesh per level in the ramp colour of
  that level, the slice one mesh per colour bin on the plane, both empty
  where the field is undefined;
* the Volume tab computes the field from the window's analysis, the window
  puts the overlays on the scene with the field's colour bar and keeps
  them across a rebuild, and the MD highlight rule draws the same over a
  frame, isolated to its window, atom and slab.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from facet.core import (basicity as B, basicity_field as F, bv,
                        coordination, readers, volume)

EXAMPLES = Path(__file__).resolve().parent.parent / "facet" / "data" / "examples"
QUARTZ = EXAMPLES / "quartz_SiO2_cod9013321.cif"
BIPO4 = EXAMPLES / "bismuth_phosphate_BiPO4_cod9008088.cif"


@pytest.fixture(scope="module")
def bipo4():
    s = readers.read(str(BIPO4))
    full = coordination.analyse_structure(s, bv.DEFAULT, cations_only=False)
    return s, full


@pytest.fixture(scope="module")
def field(bipo4):
    s, full = bipo4
    return F.basicity_grid(s, full, "pauling", probe="Pb", probe_ox=2,
                           resolution=0.3)


# -- the field --------------------------------------------------------------------

def test_the_field_reads_each_anions_lambda_at_the_anion(bipo4, field):
    s, full = bipo4
    sb = B.structure_basicity(s, full, "pauling")
    assert field.n_defined == field.n_points          # a dense oxide: no gap
    values = {index: m.value for index, m in sb.anions.items()}
    for index, m in sb.anions.items():
        atoms = [a for a in s.atoms if a.site_index == index and a.in_home_cell]
        # read off the grid between its nodes (0.3 Å), so to the grid's own
        # blur: within 2e-3 of the anion's lambda, and far closer to it
        # than to the other site's
        at = field.grid.at_cartesian(np.array([a.cart for a in atoms]))
        assert np.allclose(at, m.value, atol=2e-3)
        other = [v for k, v in values.items() if k != index]
        if other:
            assert (np.abs(at - m.value) < 0.1 * np.abs(m.value - other[0])).all()
    lo, hi = field.range
    # the grid's extremes are the anions' values, to where a node falls
    assert lo == pytest.approx(min(values.values()), abs=2e-3)
    assert hi == pytest.approx(max(values.values()), abs=2e-3)
    assert lo >= min(values.values()) - 1e-9
    assert hi <= max(values.values()) + 1e-9
    assert field.legend()[0].startswith("basicity λ sensed by Pb2+")
    assert "pauling" in field.notes[0] and "0.2 v.u." in field.notes[0]


def test_one_anion_site_gives_a_flat_field():
    s = readers.read(str(QUARTZ))
    full = coordination.analyse_structure(s, bv.DEFAULT, cations_only=False)
    f = F.basicity_grid(s, full, "pauling", probe="Na", probe_ox=1,
                        resolution=0.3)
    lo, hi = f.range
    assert hi - lo < 1e-9
    (o,) = B.structure_basicity(s, full, "pauling").anions.values()
    assert lo == pytest.approx(o.value)


def test_the_valence_grid_is_the_bond_valence_map(bipo4, field):
    s, _ = bipo4
    g = volume.bond_valence_grid(s, "Pb", 2, resolution=0.3)
    assert g.shape == field.valence.shape
    assert np.allclose(g.values, field.valence.values, atol=1e-12)


def test_the_floor_leaves_the_far_points_undefined(bipo4):
    s, full = bipo4
    high = F.basicity_grid(s, full, "pauling", probe="Pb", probe_ox=2,
                           resolution=0.3, floor_vu=3.0)
    assert 0 < high.n_defined < high.n_points
    undefined = ~np.isfinite(high.grid.values)
    assert (high.valence.values[undefined] < 3.0).all()
    assert (high.valence.values[~undefined] >= 3.0).all()
    with pytest.raises(ValueError):
        F.basicity_grid(s, full, "pauling", probe="Pb", probe_ox=0)


def test_a_frame_that_tiles_the_crystal_gives_the_same_field(bipo4):
    from facet.core import bulk, md_model

    s, full = bipo4
    crystal = F.basicity_grid(s, full, "pauling", probe="Na", probe_ox=1,
                              resolution=0.3)
    frame, _ = md_model.supercell_frame(s, (2, 2, 2))
    ox = md_model.model_oxidation(frame.species)
    ox_atom = ox.per_atom(frame.elements)
    table, _ = bulk.analyse_frame(frame, ox_atom, bv.DEFAULT, v_list_vu=0.03)
    lam = B.frame_basicity(table, "pauling", v_bond=0.075).value
    tiled = F.basicity_landscape(frame, ox_atom, lam, probe="Na", probe_ox=1,
                                 grid_spacing_ang=0.3, r_cut_ang=6.0)
    rng = np.random.default_rng(1)
    points = rng.uniform(0.0, min(s.cell.lengths), (40, 3))
    a = crystal.grid.at_cartesian(points)
    b = tiled.grid.at_cartesian(points)
    # the frame's anions carry the per-atom lambda of the same site, and the
    # two sums differ only in the cutoff (rmax 6 Å both) and the grids
    assert np.nanmax(np.abs(a - b)) < 2e-3
    assert tiled.n_defined == tiled.n_points
    assert "0.3 Å grid" in tiled.notes[0]


# -- isolating ----------------------------------------------------------------------

def test_the_masks_keep_exactly_what_they_say(field, bipo4):
    s, _ = bipo4
    grid = field.grid
    lo, hi = field.range
    mid = 0.5 * (lo + hi)
    window = F.window_mask(grid, lo, mid)
    assert window.shape == grid.shape
    inside = grid.values[window]
    assert (inside >= lo).all() and (inside <= mid).all()
    assert (grid.values[~window] > mid).all()
    centre = s.atoms[0].cart
    sphere = F.sphere_mask(grid, centre, 2.0)
    cart = (np.asarray(s.cell.orth) @ F._grid_fractional(grid).T).T
    # a point the mask keeps is within 2 Å of the centre or one of its images
    frac = F._grid_fractional(grid)
    c = np.linalg.inv(np.asarray(s.cell.orth)) @ centre
    d = frac - c
    d -= np.round(d)
    dist = np.linalg.norm((np.asarray(s.cell.orth) @ d.T).T, axis=1)
    assert np.array_equal(sphere.ravel(), dist <= 2.0)
    assert 0 < sphere.sum() < sphere.size
    normal = np.array([0.0, 0.0, 1.0])
    slab = F.slab_mask(grid, normal, 1.0, 3.0)
    along = cart @ normal
    assert np.array_equal(slab.ravel(), (along >= 1.0) & (along <= 3.0))
    axis = F.axis_slab_mask(grid, 2, 0.5, 2.0)
    assert 0 < axis.sum() < axis.size
    cut = F.restricted(grid, window)
    assert np.isfinite(cut.values).sum() == window.sum()
    assert np.array_equal(np.isfinite(cut.values), window)
    assert cut.name == grid.name


# -- drawing -----------------------------------------------------------------------

def test_the_gradient_is_one_ramp_coloured_mesh_per_level(field):
    from facet.core import theme as theme_mod

    lo, hi = field.range
    levels = F.gradient_levels(lo, hi, 5)
    assert len(levels) == 5 and lo < levels[0] < levels[-1] < hi
    assert F.gradient_levels(0.4, 0.4, 5) == [0.4]
    meshes = F.gradient_meshes(field.grid, levels, lo, hi, alpha=0.3,
                               label="g")
    assert len(meshes) == 5
    for level, mesh in zip(levels, meshes):
        assert mesh.label == "g" and mesh.alpha == pytest.approx(0.3)
        assert mesh.n_triangles > 0
        expected = theme_mod.ramp((level - lo) / (hi - lo))
        assert np.allclose(mesh.color, expected, atol=1e-9)
        assert np.isfinite(mesh.normals).all()
        # every vertex of a level surface sits on its level, to the
        # difference between the edge interpolation that placed it and the
        # trilinear one that reads it back (a few % of the range)
        at = field.grid.at_cartesian(np.asarray(mesh.vertices, float))
        deviation = np.abs(at - level)
        assert np.nanmax(deviation) < 0.1 * (hi - lo) + 1e-6
        assert np.nanmedian(deviation) < 0.02 * (hi - lo) + 1e-6
    # a window leaves the outer levels without a surface
    cut = F.restricted(field.grid, F.window_mask(field.grid, lo, levels[1]))
    fewer = F.gradient_meshes(cut, levels, lo, hi)
    assert len(fewer) < len(meshes)


def test_the_slice_is_one_mesh_per_colour_bin_on_the_plane(field, bipo4):
    s, _ = bipo4
    lo, hi = field.range
    origin = np.zeros(3)
    u, v = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    meshes = F.slice_meshes(field.grid, origin, u, v, 8.0, 20, lo, hi,
                            alpha=0.5, bins=6, label="s")
    assert 1 <= len(meshes) <= 6
    total = sum(m.n_triangles for m in meshes)
    assert total == 2 * 19 * 19                     # every cell is defined
    for m in meshes:
        assert m.label == "s" and m.alpha == pytest.approx(0.5)
        assert np.allclose(np.asarray(m.vertices)[:, 2], 0.0, atol=1e-9)
        assert np.allclose(np.asarray(m.normals), [0.0, 0.0, 1.0])
    # undefined cells are left out of the slice
    cut = F.restricted(field.grid, F.window_mask(field.grid, lo,
                                                 0.5 * (lo + hi)))
    fewer = F.slice_meshes(cut, origin, u, v, 8.0, 20, lo, hi, bins=6)
    assert 0 < sum(m.n_triangles for m in fewer) < total


def test_a_field_that_varies_everywhere_is_meshed_to_the_budget_in_seconds():
    """A glass basicity field varies at the ångström scale, so every level
    crosses a large share of the cubes: 3.1 million triangles at full
    resolution on a 55 Å box at 0.5 Å, minutes of marching, no cancel.
    The plan counts the cubes and coarsens; the whole gradient stays within
    the budget and comes back in seconds."""
    import time

    from facet.core.structure import Cell

    rng = np.random.default_rng(7)
    n = 64
    values = rng.uniform(0.4, 1.0, (n, n, n))
    cell = Cell(32.0, 32.0, 32.0, 90.0, 90.0, 90.0, np.eye(3) * 32.0)
    grid = volume.Grid(values, cell, name="noise")
    levels = F.gradient_levels(0.4, 1.0, 8)
    # at full resolution the surfaces would be millions of triangles
    assert F.straddling_cubes(values, levels[3], 1) > 0.5 * (n - 1) ** 3
    step, predicted = F.gradient_plan(values, levels)
    assert step > 1 and predicted <= F.TRIANGLE_BUDGET
    clock = time.perf_counter()
    meshes = F.gradient_meshes(grid, levels, 0.4, 1.0)
    seconds = time.perf_counter() - clock
    total = sum(m.n_triangles for m in meshes)
    assert 0 < total <= 1.5 * F.TRIANGLE_BUDGET
    assert seconds < 20.0, seconds
    # a cube with an undefined corner is never walked
    holed = values.copy()
    holed[: n // 2] = np.nan
    assert F.straddling_cubes(holed, levels[3], 1) < F.straddling_cubes(
        values, levels[3], 1)


def test_the_cancel_hook_stops_the_field_and_the_gradient(bipo4):
    from facet.core import bulk, md_channels, md_model

    s, full = bipo4
    frame, _ = md_model.supercell_frame(s, (2, 2, 2))
    ox = md_model.model_oxidation(frame.species)
    ox_atom = ox.per_atom(frame.elements)
    table, _ = bulk.analyse_frame(frame, ox_atom, bv.DEFAULT, v_list_vu=0.03)
    lam = B.frame_basicity(table, "pauling", v_bond=0.075).value
    with pytest.raises(md_channels.AnalysisCancelled):
        F.basicity_landscape(frame, ox_atom, lam, probe="Na", probe_ox=1,
                             grid_spacing_ang=0.5, r_cut_ang=6.0,
                             cancelled=lambda: True)
    fld = F.basicity_landscape(frame, ox_atom, lam, probe="Na", probe_ox=1,
                               grid_spacing_ang=0.5, r_cut_ang=6.0,
                               cancelled=lambda: False)
    lo, hi = fld.range
    with pytest.raises(md_channels.AnalysisCancelled):
        F.gradient_meshes(fld.grid, F.gradient_levels(lo, hi, 3), lo, hi,
                          cancelled=lambda: True)
    n_points, seconds = F.landscape_seconds(frame, ox_atom, 0.5, 6.0)
    assert n_points == fld.n_points and seconds > 0


# -- the panel and the window -----------------------------------------------------

def test_the_volume_tab_builds_the_field_and_asks_for_the_overlays(qapp):
    from conftest import dispose
    from facet.ui.volume_panel import VolumePanel

    s = readers.read(str(BIPO4))
    full = coordination.analyse_structure(s, bv.DEFAULT, cations_only=False)
    panel = VolumePanel()
    try:
        specs = []
        panel.fieldOverlaysChanged.connect(specs.append)
        panel.set_context(s, bv.DEFAULT,
                          basicity_source=lambda: (full, "pauling"))
        panel.kind.setCurrentIndex(panel.kind.findData("basicity"))
        assert panel.floor.isEnabled() and panel.compute_button.isEnabled()
        panel.probe_element.setCurrentText("Pb")
        panel.probe_ox.setValue(2)
        panel.resolution.setValue(0.4)
        panel.compute()
        assert panel.field is not None and panel.grid is panel.field.grid
        lo, hi = panel.field.range
        assert panel.window_lo.value() == pytest.approx(lo, abs=1e-4)
        assert panel.window_hi.value() == pytest.approx(hi, abs=1e-4)
        spec = panel.field_display()
        assert spec is not None and not spec.wanted
        panel.show_gradient.setChecked(True)
        spec = specs[-1]
        assert spec.wanted and spec.show_gradient and spec.field is panel.field
        panel.use_window.setChecked(True)
        panel.window_hi.setValue(0.5 * (lo + hi))
        assert specs[-1].window[1] == pytest.approx(0.5 * (lo + hi), abs=1e-4)
        panel.clear()
        assert specs[-1] is None and panel.field is None
    finally:
        dispose(panel)


def test_the_window_draws_the_gradient_and_the_slice_with_a_legend(qapp):
    from conftest import dispose
    from facet.ui import preview

    win = preview.PreviewWindow()
    try:
        win.load_many([str(BIPO4)])
        qapp.processEvents()
        panel = win.volume_panel
        panel.kind.setCurrentIndex(panel.kind.findData("basicity"))
        panel.probe_element.setCurrentText("Pb")
        panel.probe_ox.setValue(2)
        panel.resolution.setValue(0.4)
        panel.compute()
        qapp.processEvents()
        panel.show_gradient.setChecked(True)
        qapp.processEvents()
        labels = [m.label for m in win.scene.overlay_meshes]
        assert labels.count("field gradient") >= 2
        assert "field slice" not in labels
        title, lo, hi, _ = win.scene.value_legend
        assert title.startswith("basicity sensed by Pb2+")
        assert (lo, hi) == pytest.approx(panel.field.range, abs=1e-9)
        panel.show_slice.setChecked(True)
        qapp.processEvents()
        assert "field slice" in [m.label for m in win.scene.overlay_meshes]
        # a rebuild of the scene carries the overlays again
        win._rebuild()
        qapp.processEvents()
        labels = [m.label for m in win.scene.overlay_meshes]
        assert "field gradient" in labels and "field slice" in labels
        assert win.scene.value_legend[0].startswith("basicity sensed by")
        # isolated to a sphere about the selected site: fewer triangles
        before = sum(m.n_triangles for m in win.scene.overlay_meshes)
        entry = win.project.current
        entry.selected_site = win.project.results_for(entry)[0].site_index
        panel.region.setCurrentIndex(panel.region.findData("site"))
        qapp.processEvents()
        after = sum(m.n_triangles for m in win.scene.overlay_meshes)
        assert 0 < after < before
        # off again: nothing of the field on the scene, the atom legend back
        panel.show_gradient.setChecked(False)
        panel.show_slice.setChecked(False)
        qapp.processEvents()
        assert not [m for m in win.scene.overlay_meshes
                    if m.label.startswith("field")]
        assert win.scene.value_legend is None
    finally:
        dispose(win)


def test_the_md_rule_draws_the_map_over_a_frame_and_isolates_it(qapp):
    from facet.core import bulk, md_model
    from facet.ui.md_highlight import (BasicityMapRow, FrameData, Rule,
                                       apply_rules)
    from facet.ui.md_scene import build_md_scene

    s = readers.read(str(BIPO4))
    frame, _ = md_model.supercell_frame(s, (2, 2, 2))
    ox = md_model.model_oxidation(frame.species)
    ox_atom = ox.per_atom(frame.elements)
    table, _ = bulk.analyse_frame(frame, ox_atom, bv.DEFAULT, v_list_vu=0.03)
    fd = FrameData.from_frame(frame, table, v_bond=0.075, formers=("P",),
                              ox_atom=ox_atom)
    scene = build_md_scene(frame, table, v_bond=0.075, v_list=0.03)
    rule = Rule("basicity-map", True, probe="Pb", grid_spacing_ang=0.5,
                r_cut_ang=6.0, levels=4, alpha=0.2)
    outcome = apply_rules(scene, fd, [rule])
    assert not outcome.problems, outcome.problems
    labels = [m.label for m in scene.overlay_meshes]
    assert labels.count("basicity") == 4 and "basicity slice" not in labels
    assert scene.value_legend[0].startswith("basicity λ sensed by Pb2+")
    assert "basicity sensed by Pb2+" in outcome.note
    # a slice, a window and the slab
    fd.slab = (2, 0.5, 4.0)
    fld = fd.basicity_field("Pb", 0.5, 6.0, 0.2)
    lo, hi = fld.range
    cut = Rule("basicity-map", True, probe="Pb", grid_spacing_ang=0.5,
               r_cut_ang=6.0, levels=4, alpha=0.2, window=(lo, 0.5 * (lo + hi)),
               region="slab", slice_axis=2, slice_fraction=0.5)
    scene2 = build_md_scene(frame, table, v_bond=0.075, v_list=0.03)
    outcome2 = apply_rules(scene2, fd, [cut])
    assert not outcome2.problems, outcome2.problems
    labels2 = [m.label for m in scene2.overlay_meshes]
    assert "basicity slice" in labels2
    assert sum(m.n_triangles for m in scene2.overlay_meshes
               if m.label == "basicity") < sum(
        m.n_triangles for m in scene.overlay_meshes if m.label == "basicity")
    assert "inside the slab along c" in outcome2.note
    assert scene2.value_legend[1:3] == pytest.approx((lo, 0.5 * (lo + hi)))
    # the row builds that rule and reads it back from a saved set
    row = BasicityMapRow()
    row.set_frame_data(fd)
    assert "Pb" in [row.probe.itemText(i) for i in range(row.probe.count())]
    from dataclasses import asdict
    row.load_spec(asdict(cut))
    back = row.rule()
    assert back.probe == "Pb"
    assert back.window == pytest.approx(cut.window, abs=1e-4)   # the spins' 4 decimals
    assert back.region == "slab" and back.slice_axis == 2
    row.deleteLater()
    # a probe without a parameter is reported, not raised
    bad = Rule("basicity-map", True, probe="Xx", grid_spacing_ang=0.5,
               r_cut_ang=6.0)
    scene3 = build_md_scene(frame, table, v_bond=0.075, v_list=0.03)
    outcome3 = apply_rules(scene3, fd, [bad])
    assert outcome3.problems and "basicity map" in outcome3.problems[0]
