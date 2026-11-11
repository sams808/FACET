"""The ways of seeing the microscopic basicity added in 1.3.0, and the flaws
an audit of the shared build found, each pinned.

* the regional map: the field averaged over a Gaussian narrows towards its
  mean, keeps its mean and leaves the undefined points undefined;
* the regions: two levels that bound the stated share at each end;
* the field repeats over the cell range the atoms are drawn over (it was
  drawn in the home cell only, and so was the single isosurface);
* an over-bonded anion says what its bonds carry (two 1930s structures of
  the local collection read lambda = -3.3 with no word of why);
* every atom coloured: a cation's value is the crystal path's environment
  basicity, atom for atom;
* a threshold move does not reuse the field of another threshold (the
  landscapes are handed on between thresholds, and lambda is not theirs);
* a change of opacity marches nothing again;
* a session keeps the scale, and a field on the previous scale is cleared.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from facet.core import (basicity as B, basicity_field as F, bv,
                        coordination, readers)

EXAMPLES = Path(__file__).resolve().parent.parent / "facet" / "data" / "examples"
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


@pytest.fixture(scope="module")
def frame_data(bipo4):
    from facet.core import bulk, md_model
    from facet.ui.md_highlight import FrameData

    s, _ = bipo4
    frame, _ = md_model.supercell_frame(s, (2, 2, 2))
    ox = md_model.model_oxidation(frame.species)
    ox_atom = ox.per_atom(frame.elements)
    table, _ = bulk.analyse_frame(frame, ox_atom, bv.DEFAULT, v_list_vu=0.03)
    return frame, table, ox_atom, FrameData.from_frame(
        frame, table, v_bond=0.075, formers=("P",), ox_atom=ox_atom)


def test_smoothing_gives_the_regional_map_and_keeps_the_undefined(field):
    assert F.smoothed(field, 0.0) is field
    smooth = F.smoothed(field, 1.5)
    lo, hi = field.range
    slo, shi = smooth.range
    assert lo < slo < shi < hi                       # an average narrows
    assert np.nanmean(smooth.grid.values) == pytest.approx(
        np.nanmean(field.grid.values), abs=2e-3)
    assert "sigma 1.5 Å" in smooth.notes[-1] and smooth.probe == field.probe
    holed = F.BasicityField(
        F.restricted(field.grid, F.window_mask(field.grid, lo, 0.5 * (lo + hi))),
        field.valence, field.probe, field.probe_ox, field.scale,
        field.floor_vu, 0, field.n_points)
    again = F.smoothed(holed, 1.0)
    assert np.array_equal(np.isfinite(again.grid.values),
                          np.isfinite(holed.grid.values))
    with pytest.raises(ValueError):
        F.smoothed(field, -1.0)


def test_region_levels_and_tiling(bipo4, field):
    s, _ = bipo4
    values = np.asarray(field.grid.values)
    low, high = F.region_levels(values, 15)
    assert (values <= low).mean() == pytest.approx(0.15, abs=0.02)
    assert (values >= high).mean() == pytest.approx(0.15, abs=0.02)
    assert F.region_levels(np.full((4, 4, 4), np.nan), 15) == []
    with pytest.raises(ValueError):
        F.region_levels(values, 60)
    lo, hi = field.range
    one = F.gradient_meshes(field.grid, [low, high], lo, hi)
    many = F.tiled(one, s.cell, (2, 2, 2))
    assert F.tiled(one, s.cell, (1, 1, 1)) == one
    assert sum(m.n_triangles for m in many) == 8 * sum(m.n_triangles for m in one)
    span_one = np.ptp(np.vstack([m.vertices for m in one]), axis=0)
    span_many = np.ptp(np.vstack([m.vertices for m in many]), axis=0)
    assert (span_many > 1.5 * span_one).all()


def test_an_overbonded_anion_says_what_its_bonds_carry():
    from test_basicity import _anion

    twice = B.micro_basicity(_anion([("Si", 4, 4)] * 4), "pauling")   # 4 v.u.
    assert twice.value < 0.0
    assert any("4.00 v.u. against a formal charge of 2" in n
               for n in twice.notes)
    usual = B.micro_basicity(_anion([("Si", 4, 4)] * 2), "pauling")
    assert not usual.notes


def test_every_atom_is_coloured_and_the_cations_read_their_anions(frame_data):
    from facet.core import md_site, theme as theme_mod
    from facet.ui.md_highlight import Rule, _keep_of_rule, colour_atoms
    from facet.ui.md_scene import build_md_scene

    frame, table, _ox, fd = frame_data
    everywhere = fd.basicity_everywhere()
    assert np.isfinite(everywhere).all()
    # a cation's value is the crystal path's EnvironmentBasicity, atom for atom
    row = int(np.flatnonzero(np.asarray(frame.elements) == "Bi")[0])
    site = md_site.site_result_for_atom(frame, table, row, v_bond=0.075)
    anions = {int(c.site_index): B.micro_basicity(md_site.site_result_for_atom(
        frame, table, int(c.site_index), v_bond=0.075), "pauling")
        for c in site.contacts if c.has_valence and c.valence > 0.075}
    env = B.environment_basicity(site, anions, "pauling")
    assert everywhere[row] == pytest.approx(env.value, abs=1e-12)
    theme = theme_mod.Theme()
    scene = build_md_scene(frame, table, theme=theme, v_bond=0.075, v_list=0.03)
    assert colour_atoms(scene, fd, "basicity about every atom", theme) == ""
    colours = np.asarray(scene.atom_color)
    sites = np.asarray(scene.atom_site)
    assert not np.allclose(colours[sites == row][0], theme.element_color("Bi"))
    assert "mean λ about the cations" in scene.value_legend[0]
    # the descriptor isolates the oxygens above a value
    lam = fd.basicity_values()
    cut = float(np.nanmedian(lam))
    keep = _keep_of_rule(fd, Rule("show-only", True, element="O",
                                  descriptor="microscopic basicity",
                                  sense="≥", value=cut))
    assert keep.sum() == int((lam >= cut).sum()) > 0
    assert (np.asarray(frame.elements)[keep] == "O").all()


def test_the_map_follows_the_threshold_and_keeps_its_triangles(frame_data):
    from facet.ui.md_highlight import FrameData, Rule, apply_rules
    from facet.ui.md_scene import build_md_scene

    frame, table, ox_atom, fd = frame_data

    def rule(alpha):
        return Rule("basicity-map", True, probe="Pb", grid_spacing_ang=0.6,
                    r_cut_ang=6.0, levels=4, alpha=alpha, smooth_ang=1.0,
                    regions_percent=15)

    scene = build_md_scene(frame, table, v_bond=0.075, v_list=0.03)
    out = apply_rules(scene, fd, [rule(0.2)])
    assert not out.problems, out.problems
    meshes = [m for m in scene.overlay_meshes if m.label == "basicity"]
    assert len(meshes) == 2
    assert "most acidic and the most basic 15 %" in out.note
    assert "averaged over 1 Å" in out.note
    # another opacity: the same triangles, nothing marched again
    n_kept = len(fd._basicity_meshes)
    again = build_md_scene(frame, table, v_bond=0.075, v_list=0.03)
    apply_rules(again, fd, [rule(0.6)])
    assert len(fd._basicity_meshes) == n_kept == 1
    other = [m for m in again.overlay_meshes if m.label == "basicity"]
    assert other[0].alpha == pytest.approx(0.6)
    assert sum(m.n_triangles for m in other) == sum(m.n_triangles for m in meshes)
    # another threshold: the landscapes are handed on, the field is not reused
    moved = FrameData.from_frame(frame, table, v_bond=0.5, formers=("P",),
                                 ox_atom=ox_atom)
    moved._landscapes = fd._landscapes
    a = fd.basicity_field("Pb", 0.6, 6.0, 0.2)
    b = moved.basicity_field("Pb", 0.6, 6.0, 0.2)
    assert a is not b and a.range != b.range


def test_the_window_tiles_the_field_and_a_session_keeps_the_scale(qapp, tmp_path):
    import json

    from conftest import dispose
    from facet.ui import preview

    win = preview.PreviewWindow()
    other = None
    try:
        win.load_many([str(BIPO4)])
        qapp.processEvents()
        panel = win.volume_panel
        panel.kind.setCurrentIndex(panel.kind.findData("basicity"))
        panel.probe_element.setCurrentText("Pb")
        panel.probe_ox.setValue(2)
        panel.resolution.setValue(0.5)
        panel.smooth.setValue(0.5)
        panel.compute()
        assert "sigma 0.5 Å" in panel.grid.name
        panel.use_regions.setChecked(True)
        panel.show_gradient.setChecked(True)
        qapp.processEvents()

        def span():
            found = [m for m in win.scene.overlay_meshes
                     if m.label == "field gradient"]
            return len(found), np.ptp(np.vstack([m.vertices for m in found]),
                                      axis=0)

        n, one = span()
        assert n == 2
        for box in win.range_boxes:
            box.setValue(2)
        win._rebuild()
        qapp.processEvents()
        _n, two = span()
        assert (two > 1.5 * one).all()
        # the scale: on the theme, in the session, restored; a field on the
        # previous scale is cleared rather than left under the new name
        win.utilities.set_basicity_scale("lebouteiller")
        qapp.processEvents()
        assert win.theme.basicity_scale == "lebouteiller"
        assert panel.field is None and panel.grid is None
        path = tmp_path / "s.facet"
        win.save_session(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        other = preview.PreviewWindow()
        other.restore_session(data)
        qapp.processEvents()
        assert other.utilities.basicity_scale() == "lebouteiller"
    finally:
        dispose(win, other)
