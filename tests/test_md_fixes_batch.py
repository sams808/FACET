"""Three things a glass model showed were off in 1.0.0, each pinned
against something outside the code that produced it:

* the angle arcs of an isolated MD environment: the cluster's bond
  endpoints are renumbered with the atoms, so the arcs find the bonds
  (they found none: the endpoints still named the whole frame's atoms);
* voids: Shannon's ionic radii by the model's oxidation states, the
  sphere moved to the maximum of the clearance and checked against a
  brute force over every image, one sphere per maximum, a drawing budget;
* channels by charge: the target the sum is matched to can be the mean of
  the sum at the probe's own atoms, so the regions pass through the
  modifier sites; the probe sums equal the landscape at those atoms.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from facet.core import bv, md_channels, md_model, md_order, readers

CRYSTALS = Path(__file__).resolve().parent / "data" / "crystals"


def _glassy_frame(reps=(3, 3, 3), jitter=0.08, seed=5):
    structure = readers.read(CRYSTALS / "quartz_SiO2_cod9013321.cif")
    frame, _ = md_model.supercell_frame(structure, reps)
    rng = np.random.default_rng(seed)
    cart = frame.frac @ frame.box_ang + rng.normal(0.0, jitter, (frame.n_atoms, 3))
    return md_model.frame_from_arrays(frame.elements, cart, box_ang=frame.box_ang,
                                      atom_id=frame.atom_id)


# -- voids -----------------------------------------------------------------------------

def test_ionic_radii_follow_shannon_and_the_models_oxidation_states():
    frame = _glassy_frame((2, 2, 2))
    ox = md_model.model_oxidation(frame.species).per_atom(frame.elements)
    radii, source = md_order.ionic_radii_ang(frame, ox)
    assert radii == {"O": 1.40, "Si": 0.40}
    assert source.startswith("Shannon (1976)") and "van der Waals" not in source
    # an ion the table lacks falls back to the van der Waals radius, and says so
    table, source = md_order.ionic_radii_ang(["Xe"], None)
    assert "no ionic radius tabulated for Xe" in source and table["Xe"] > 1.5
    assert md_order.IONIC_RADII_ANG[("Ca", 2)] == 1.00
    assert md_order.IONIC_RADII_ANG[("Na", 1)] == 1.02


def test_refined_voids_equal_a_brute_force_over_every_image_and_never_lose_clearance():
    frame = _glassy_frame((2, 2, 2))
    ox = md_model.model_oxidation(frame.species).per_atom(frame.elements)
    radii, source = md_order.ionic_radii_ang(frame, ox)
    voids = md_order.empty_spheres(frame, radii, radii_source=source)
    centre = voids.centre_frac @ frame.box_ang
    shifts = np.array(list(itertools.product(range(-1, 2), repeat=3)))
    images = ((frame.frac[None] + shifts[:, None]).reshape(-1, 3) @ frame.box_ang)
    r_image = np.tile([radii[s] for s in frame.elements], len(shifts))
    dist = np.linalg.norm(centre[:, None, :] - images[None, :, :], axis=2)
    brute = (dist - r_image).min(axis=1)
    assert np.allclose(voids.radius_ang, brute, rtol=0, atol=1e-9)
    assert (voids.radius_ang >= voids.seed_radius_ang - 1e-9).all()
    assert (voids.moved_ang <= voids.circumradius_ang + 1e-9).all()
    assert voids.representative.any() and voids.n_voids < len(voids)
    # the largest void is larger than any seed, and with ionic radii every
    # sphere has room (with van der Waals radii most do not)
    assert voids.radius_ang.max() >= voids.seed_radius_ang.max()
    vdw = md_order.empty_spheres(frame, md_order.vdw_radii_ang(frame),
                                 radii_source=md_order.VDW_RADII_SOURCE)
    assert voids.radius_ang.mean() > vdw.radius_ang.mean() + 0.1


def test_the_void_count_is_translation_invariant():
    frame = _glassy_frame((2, 2, 2))
    ox = md_model.model_oxidation(frame.species).per_atom(frame.elements)
    radii, source = md_order.ionic_radii_ang(frame, ox)
    moved = md_model.frame_from_arrays(frame.elements, frac=frame.frac + 0.21,
                                       box_ang=frame.box_ang)
    a = md_order.empty_spheres(frame, radii, radii_source=source)
    b = md_order.empty_spheres(moved, radii, radii_source=source)
    ra = np.sort(a.radius_ang[a.representative])
    rb = np.sort(b.radius_ang[b.representative])
    assert a.n_voids == b.n_voids
    assert np.allclose(ra, rb, atol=2e-3)


def test_the_voids_rule_takes_the_radii_and_draws_within_the_budget(qapp):
    from facet.ui import md_highlight as mh
    frame = _glassy_frame((3, 3, 3))
    ox = md_model.model_oxidation(frame.species).per_atom(frame.elements)
    from facet.core import bulk
    table, _ = bulk.analyse_frame(frame, ox, bv.DEFAULT, v_bond_vu=0.075,
                                  v_list_vu=0.02)
    fd = mh.FrameData.from_frame(frame, table, k=0, v_bond=0.075,
                                 formers=frozenset({"Si"}), ox_atom=ox)
    ionic = fd.spheres_with("ionic")
    vdw = fd.spheres_with("vdw")
    assert fd.spheres is ionic and fd.spheres_with("ionic") is ionic
    assert ionic.radii_source.startswith("Shannon")
    assert (ionic.radius_ang > 0).sum() > (vdw.radius_ang > 0).sum()
    from facet.ui.md_scene import build_md_scene
    scene = build_md_scene(frame)
    mh.VOID_DRAW_BUDGET, saved = 5, mh.VOID_DRAW_BUDGET
    try:
        result = mh.apply_rules(scene, fd, [mh.Rule("voids", min_volume_ang3=0.01,
                                                     radii="ionic")], dim_rest=True)
        assert result.n_spheres > 5
        mesh = [m for m in scene.overlay_meshes if m.label == "voids"][0]
        assert mesh.n_parts == 5
        assert "the largest 5 drawn" in result.note
        assert "Shannon" in result.note
    finally:
        mh.VOID_DRAW_BUDGET = saved
    row = mh.VoidsRow()
    try:
        row.radii.setCurrentIndex(1)
        assert row.rule().radii == "vdw"
        page = mh.HighlightPage()
        try:
            page.from_dict({"rules": [{"kind": "voids", "min_volume_ang3": 2.0,
                                       "radii": "none"}]})
            assert page.rules()[0].radii == "none"
            assert page.minimumSizeHint().width() <= 470
        finally:
            page.deleteLater()
    finally:
        row.deleteLater()


# -- channels by charge ----------------------------------------------------------------

def test_the_frame_target_is_the_mean_of_the_probe_sums():
    frame = _glassy_frame((2, 2, 2))
    ox = md_model.model_oxidation(frame.species).per_atom(frame.elements)
    sums = md_channels.probe_sums(frame, ox, "Si", 4, r_cut_ang=4.0)
    assert sums.shape == ((np.asarray(frame.elements) == "Si").sum(),)
    formal = md_channels.bv_landscape(frame, ox, "Si", 4, grid_spacing_ang=0.6,
                                      r_cut_ang=4.0, target="formal")
    mean = md_channels.bv_landscape(frame, ox, "Si", 4, grid_spacing_ang=0.6,
                                    r_cut_ang=4.0, target="frame")
    assert formal.target_vu == 4.0 and formal.target_mode == "formal"
    assert mean.target_mode == "frame"
    assert mean.target_vu == pytest.approx(float(sums.mean()))
    assert np.allclose(mean.probe_sums_vu, sums)
    assert np.allclose(formal.bvs_vu, mean.bvs_vu)
    assert np.allclose(mean.mismatch_vu, np.abs(formal.bvs_vu - mean.target_vu))
    assert any("mean of V at the" in n for n in mean.notes)
    given = md_channels.bv_landscape(frame, ox, "Si", 4, grid_spacing_ang=0.6,
                                     r_cut_ang=4.0, target=3.5)
    assert given.target_vu == 3.5 and given.target_mode == "given"
    with pytest.raises(ValueError):
        md_channels.bv_landscape(frame, ox, "Si", 4, grid_spacing_ang=0.6,
                                 r_cut_ang=4.0, target="nope")
    # a probe the frame lacks: the formal valence, said in the notes
    none = md_channels.bv_landscape(frame, ox, "Na", 1, grid_spacing_ang=0.6,
                                    r_cut_ang=4.0, target="frame")
    assert none.target_mode == "formal" and any("holds no Na" in n for n in none.notes)


def test_the_frame_target_puts_the_probe_atoms_inside_the_regions():
    """Against the formal valence a jittered quartz's own Si, which sum to
    about 4.2, are missed at a tight mismatch; against the frame's mean
    they are inside."""
    frame = _glassy_frame((2, 2, 2), jitter=0.12, seed=9)
    ox = md_model.model_oxidation(frame.species).per_atom(frame.elements)
    sums = md_channels.probe_sums(frame, ox, "Si", 4, r_cut_ang=4.0)
    delta = float(2 * sums.std())
    si = np.asarray(frame.elements) == "Si"
    inside = {}
    for target in ("formal", "frame"):
        land = md_channels.bv_landscape(frame, ox, "Si", 4, grid_spacing_ang=0.25,
                                        r_cut_ang=4.0, target=target)
        regions = md_channels.accessible_regions(land, delta)
        inside[target] = int((np.asarray(regions.region_of_atom)[si] >= 0).sum())
    assert inside["frame"] >= inside["formal"]
    assert inside["frame"] >= 0.9 * si.sum()


def test_the_channel_row_offers_the_target_and_the_spread(qapp):
    from facet.ui import md_highlight as mh
    frame = _glassy_frame((2, 2, 2))
    ox = md_model.model_oxidation(frame.species).per_atom(frame.elements)
    from facet.core import bulk
    table, _ = bulk.analyse_frame(frame, ox, bv.DEFAULT, v_bond_vu=0.075,
                                  v_list_vu=0.02)
    fd = mh.FrameData.from_frame(frame, table, k=0, v_bond=0.075,
                                 formers=frozenset({"Si"}), ox_atom=ox)
    mean, sd, n = fd.probe_sum_spread("Si", 4.0)
    assert n == int((np.asarray(frame.elements) == "Si").sum()) and sd > 0
    row = mh.ChannelsRow()
    try:
        row.set_frame_data(fd)
        i = row.probe.findText("Si")
        assert i >= 0
        row.probe.setCurrentIndex(i)
        row.cut.setValue(4.0)
        assert row.rule().target == "frame"
        row.target.setCurrentIndex(1)
        assert row.rule().target == "formal"
        row.spread_button.click()
        assert row.threshold.value() == pytest.approx(max(0.01, round(2 * sd, 2)))
        assert "Δ set to" in row.note.text()
    finally:
        row.deleteLater()
    page = mh.HighlightPage()
    try:
        page.from_dict({"rules": [{"kind": "channels", "channel_by": "by charge",
                                   "probe": "Si", "target": "formal"}]})
        assert page.rules()[0].target == "formal"
        assert page.minimumSizeHint().width() <= 470
    finally:
        page.deleteLater()


# -- the arcs of an isolated environment -------------------------------------------

def test_an_isolated_cluster_keeps_its_bond_endpoints_numbered(qapp):
    from facet.ui import md_highlight as mh
    from facet.ui.md_scene import build_md_scene
    from facet.core import bulk
    frame = _glassy_frame((2, 2, 2))
    ox = md_model.model_oxidation(frame.species).per_atom(frame.elements)
    table, _ = bulk.analyse_frame(frame, ox, bv.DEFAULT, v_bond_vu=0.075,
                                  v_list_vu=0.02)
    scene = build_md_scene(frame, table, v_bond=0.075, v_list=0.02)
    ends = np.asarray(scene.bond_atoms, int)
    # a Si far from row 0, so the renumbering is not the identity
    si_rows = np.flatnonzero(np.asarray(frame.elements) == "Si")
    centre = int(si_rows[len(si_rows) // 2])
    keep_rows = np.zeros(frame.n_atoms, bool)
    touching = (ends[:, 0] == centre) | (ends[:, 1] == centre)
    keep_rows[np.unique(ends[touching])] = True
    keep_rows[centre] = True
    n_bonds_before = int(touching.sum())
    mh._hide(scene, keep_rows)
    here = int(np.flatnonzero(np.asarray(scene.atom_site) == centre)[0])
    ends = np.asarray(scene.bond_atoms, int)
    assert (ends >= 0).all() and (ends < scene.n_atoms).all()
    assert int(((ends[:, 0] == here) | (ends[:, 1] == here)).sum()) == n_bonds_before
    # the arcs find the bonds through the view's own lookup
    from facet.gl.view import StructureView
    view = StructureView()
    try:
        view.set_scene(scene, reframe=True)
        view.set_angle_hub(here)
        pairs = view.hub_angles()
        assert len(pairs) == n_bonds_before * (n_bonds_before - 1) // 2
    finally:
        view.deleteLater()
