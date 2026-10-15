"""The requests of 2026-10-15 on the window: only the highlight when the
rest is left out (a void rule keeps its lining atoms), a slider for how
far the rest dims, a hot map of the modifiers, the arcs of the front bonds
only, a lone-pair lobe per site from the right click, a site isolated with
its neighbours in the crystal view, labels on chosen atoms and families,
and a bond readout on a click.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import dispose

from facet.core import bv, cif, coordination, md_hotmap, md_model, readers, prefs

CRYSTALS = Path(__file__).resolve().parent / "data" / "crystals"
EXAMPLES = Path(__file__).resolve().parent.parent / "facet" / "data" / "examples"
QUARTZ = CRYSTALS / "quartz_SiO2_cod9013321.cif"
EULYTITE = EXAMPLES / "eulytite_Bi4SiO4_3_cod9012894.cif"


def _glassy_frame(reps=(3, 3, 3), jitter=0.08, seed=5):
    structure = readers.read(QUARTZ)
    frame, _ = md_model.supercell_frame(structure, reps)
    rng = np.random.default_rng(seed)
    cart = frame.frac @ frame.box_ang + rng.normal(0.0, jitter, (frame.n_atoms, 3))
    return md_model.frame_from_arrays(frame.elements, cart, box_ang=frame.box_ang,
                                      atom_id=frame.atom_id)


def _frame_data(frame, formers=frozenset({"Si"})):
    from facet.core import bulk
    from facet.ui import md_highlight as mh
    ox = md_model.model_oxidation(frame.species).per_atom(frame.elements)
    table, _ = bulk.analyse_frame(frame, ox, bv.DEFAULT, v_bond_vu=0.075,
                                  v_list_vu=0.02)
    return mh.FrameData.from_frame(frame, table, k=0, v_bond=0.075,
                                   formers=formers, ox_atom=ox), table


def _scene(fd, table):
    from facet.ui.md_scene import build_md_scene
    return build_md_scene(fd.frame, table, v_bond=0.075, v_list=0.02)


# -- the hot map -----------------------------------------------------------------------

def test_modifier_counts_and_density_integrate_to_the_atoms():
    frame = _glassy_frame((2, 2, 2))
    counts = md_hotmap.modifier_counts(frame, ["Si"], 3.5)
    assert counts.shape == (frame.n_atoms,)
    # a brute force over the images for one atom
    i = 0
    shifts = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)])
    cart = frame.frac @ frame.box_ang
    si = np.flatnonzero(np.asarray(frame.elements) == "Si")
    images = ((frame.frac[si][None] + shifts[:, None]).reshape(-1, 3) @ frame.box_ang)
    d = np.linalg.norm(images - cart[i], axis=1)
    brute = int(((d <= 3.5) & (d > 1e-9)).sum())
    assert int(counts[i]) == brute
    hot = md_hotmap.modifier_density_grid(frame, ["Si"], grid_spacing_ang=0.4,
                                          sigma_ang=1.0)
    integral = float(np.asarray(hot.grid.values).sum()) * frame.volume_ang3 / hot.grid.n_points
    assert integral == pytest.approx(hot.n_atoms, rel=2e-3)
    assert hot.mean_density == pytest.approx(hot.n_atoms / frame.volume_ang3)
    assert 0 < hot.fraction_above(1.5 * hot.mean_density) < 1
    with pytest.raises(ValueError):
        md_hotmap.modifier_counts(frame, ["Na"], 3.0)
    with pytest.raises(ValueError):
        md_hotmap.modifier_density_grid(frame, [], grid_spacing_ang=0.5, sigma_ang=1.0)


def test_the_hot_map_rule_draws_a_surface_and_the_colour_mode_counts(qapp):
    from facet.ui import md_highlight as mh
    frame = _glassy_frame((3, 3, 3))
    # O as the former, so Si are the modifiers
    fd, table = _frame_data(frame, formers=frozenset({"O"}))
    assert fd.modifiers == ("Si",)
    scene = _scene(fd, table)
    result = mh.apply_rules(scene, fd, [mh.Rule("hotmap", level_x_mean=1.2,
                                                 sigma_ang=1.2,
                                                 grid_spacing_ang=0.5)],
                            dim_rest=True, colour_by="modifier density")
    assert not result.problems, result.problems
    meshes = [m for m in scene.overlay_meshes if m.label == "hotmap"]
    assert len(meshes) == 1 and meshes[0].n_triangles > 0
    assert "hot map of Si" in result.note and "× the mean" in result.note
    assert "within 4 Å" in result.note
    # one apply replaces the surface, never doubles it
    mh.apply_rules(scene, fd, [mh.Rule("hotmap", level_x_mean=1.2, sigma_ang=1.2,
                                        grid_spacing_ang=0.5)], dim_rest=True)
    assert len([m for m in scene.overlay_meshes if m.label == "hotmap"]) == 1
    # a level above the peak draws nothing and says what the peak is
    result = mh.apply_rules(scene, fd, [mh.Rule("hotmap", level_x_mean=5.0,
                                                 sigma_ang=1.2, grid_spacing_ang=0.5)],
                            dim_rest=True)
    assert not [m for m in scene.overlay_meshes if m.label == "hotmap"]
    assert "encloses 0.0 % of the box" in result.note
    row = mh.HotmapRow()
    try:
        row.set_frame_data(fd)
        assert row.rule().elements == "" and row.rule().level_x_mean == 1.5
        row.elements.setEditText("Si, O")
        assert row.rule().elements == "Si, O"
    finally:
        row.deleteLater()
    page = mh.HighlightPage()
    try:
        page.from_dict({"rules": [{"kind": "hotmap", "level_x_mean": 2.0,
                                   "sigma_ang": 1.1, "elements": "Si"}]})
        rule = page.rules()[0]
        assert rule.kind == "hotmap" and rule.level_x_mean == 2.0 \
            and rule.sigma_ang == pytest.approx(1.1) and rule.elements == "Si"
        assert page.minimumSizeHint().width() <= 470
    finally:
        page.deleteLater()


# -- the dimming and the lining ----------------------------------------------------------

def test_a_void_rule_keeps_its_lining_when_the_rest_is_left_out(qapp):
    from facet.ui import md_highlight as mh
    frame = _glassy_frame((3, 3, 3))
    fd, table = _frame_data(frame)
    scene = _scene(fd, table)
    n_all = scene.n_atoms
    result = mh.apply_rules(scene, fd, [mh.Rule("voids", min_volume_ang3=0.3)],
                            dim_rest=False)
    assert "atoms line the drawn voids" in result.note
    assert "the rest left out" in result.note
    assert 0 < scene.n_atoms < n_all
    assert result.keep is not None and 0 < result.keep.sum() < fd.n_atoms


def test_the_dim_amount_fades_towards_the_background_as_asked(qapp):
    from facet.ui import md_highlight as mh
    from facet.core import theme as theme_mod
    frame = _glassy_frame((2, 2, 2))
    fd, table = _frame_data(frame)
    theme = theme_mod.Theme()
    bg = np.asarray(theme.background, np.float32)
    rule = mh.Rule("show-only", element="Si", descriptor="CN", sense="≥", value=0)
    outside = np.asarray(fd.elements) != "Si"
    for amount, expect in ((0.0, None), (1.0, "background")):
        scene = _scene(fd, table)
        base = np.asarray(scene.atom_color).copy()
        mh.apply_rules(scene, fd, [rule], dim_rest=True, theme=theme, dim_amount=amount)
        rows = np.asarray(scene.atom_site, int)
        faded = np.asarray(scene.atom_color)[outside[rows]]
        if expect is None:
            assert np.allclose(faded, base[outside[rows]])
        else:
            assert np.allclose(faded, bg, atol=1e-6)
    page = mh.HighlightPage()
    try:
        assert page.dim_amount() == pytest.approx(mh.DIM_DEFAULT)
        page.dim_slider.setValue(35)
        assert page.dim_amount() == pytest.approx(0.35)
        data = page.to_dict()
        assert data["dim_amount"] == pytest.approx(0.35)
        page.dim_slider.setValue(80)
        page.from_dict(data)
        assert page.dim_slider.value() == 35
    finally:
        page.deleteLater()


# -- the crystal view ------------------------------------------------------------------

@pytest.fixture
def window(qapp):
    prefs.set("tour_offered", True)
    from facet.ui.preview import PreviewWindow
    w = PreviewWindow()
    yield w
    w.close()
    dispose(w)


def test_front_bonds_only_halves_a_crowded_site(qapp, window):
    window.load(str(EULYTITE))
    bi = next(r for r in window.results if r.element == "Bi")
    scene = window.view.scene
    hub = int(np.flatnonzero(np.asarray(scene.atom_site) == bi.site_index)[0])
    view = window.view
    view.set_angle_hub(hub)
    pairs = view.hub_angles()
    assert len(pairs) == bi.cn_valence * (bi.cn_valence - 1) // 2
    assert view.angles_front_only
    front = view.front_bonds()
    bonds = {b for i, j, _a in pairs for b in (i, j)}
    assert 0 < len(front) < len(bonds)
    view.set_angles_front_only(False)
    assert not view.angles_front_only


def test_a_lone_pair_lobe_is_drawn_on_chosen_sites_only(qapp, window):
    window.load(str(EULYTITE))
    bi = next(r for r in window.results if r.element == "Bi")
    assert bi.phi > 0.1
    assert window.view.scene.vector_sites is None
    before = len(window.view.scene.overlay_meshes)
    window._toggle_lone_pair(bi.site_index)
    scene = window.view.scene
    assert scene.vector_sites == frozenset({bi.site_index})
    lobes = [m for m in scene.overlay_meshes if "vector" in (m.label or "")]
    assert len(scene.overlay_meshes) > before or lobes
    window._toggle_lone_pair(bi.site_index)
    assert not window.view.scene.vector_sites


def test_isolating_an_atom_keeps_its_shells_and_renumbers(qapp, window):
    from facet.gl.scene import isolate_atoms, neighbour_shells_of
    window.load(str(QUARTZ))
    scene = window.view.scene
    si = int(np.flatnonzero(np.asarray(scene.atom_element) == "Si")[0])
    alone = neighbour_shells_of(scene, si, 0)
    one = neighbour_shells_of(scene, si, 1)
    two = neighbour_shells_of(scene, si, 2)
    assert alone.sum() == 1 and one.sum() == 5 and two.sum() > one.sum()
    n_all = scene.n_atoms
    window._isolate(si, 1)
    s = window.view.scene
    assert s.n_atoms == 5 and s.n_bonds >= 4
    ends = np.asarray(s.bond_atoms, int)
    assert (ends >= 0).all() and (ends < s.n_atoms).all()
    window._isolate(None, 0)
    assert window.view.scene.n_atoms == n_all


def test_labels_on_chosen_atoms_and_families(qapp, window):
    from facet.gl import labels as labels_mod
    from facet.gl.labels import AtomLabel, BondLabel, LabelScope
    window.load(str(QUARTZ))
    window.atom_label_box.setCurrentIndex(window.atom_label_box.findData(AtomLabel.SITE))
    window.bond_label_box.setCurrentIndex(window.bond_label_box.findData(BondLabel.DISTANCE))
    window._on_labels()
    scene = window.view.scene
    settings = window.labels

    by_site = {r.site_index: r for r in window.results}

    def count(kind):
        if kind == "atom":
            return len(labels_mod._atom_candidates(
                scene, window.view.camera, settings, 800, 600, by_site,
                window.structure, None))
        return len(labels_mod._bond_candidates(
            scene, window.view.camera, settings, 800, 600, by_site, None))

    atoms_all, bonds_all = count("atom"), count("bond")
    assert atoms_all > 0 and bonds_all > 0
    si = next(r for r in window.results if r.element == "Si")
    window.set_label_scope(sites={si.site_index})
    assert settings.atom_scope is LabelScope.CHOSEN and settings.bond_scope is LabelScope.CHOSEN
    assert window.label_scope_button.text().startswith("on 1")
    atoms_si = count("atom")
    assert 0 < atoms_si < atoms_all
    assert 0 < count("bond") <= bonds_all
    window.set_label_scope(elements={"O"})
    assert count("atom") > 0
    window.set_label_scope()
    assert settings.atom_scope is LabelScope.ALL
    assert count("atom") == atoms_all


def test_a_click_on_a_bond_reads_out_the_parameter_and_the_share(qapp, window):
    window.load(str(QUARTZ))
    scene = window.view.scene
    si = next(r for r in window.results if r.element == "Si")
    bond = next(i for i in range(scene.n_bonds)
                if int(scene.atom_site[int(scene.bond_atoms[i, 0])]) == si.site_index)
    window._on_bond_picked(bond)
    text = window.statusBar().currentMessage()
    assert text.startswith("Bond") and "distance" in text and "valence" in text
    logged = [m[1] if isinstance(m, tuple) else str(m) for m in window.messages()]
    assert any("R0" in m and "cutoff" in m and "% of" in m for m in logged)
