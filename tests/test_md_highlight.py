"""The Highlight page of the Model workspace (facet.ui.md_highlight) and
the rules it applies to a built Scene.

What these tests pin, on a real frame (a 3 x 3 x 3 quartz supercell, 243
atoms, and the 3 000-atom Na2O-3SiO2 model where it is on this machine):

* a show-only rule keeps exactly the atoms a direct numpy count keeps, and
  "Dim the rest" shrinks and pales the others where hiding leaves them out;
* the void spheres are drawn from the frame's empty spheres, at or above
  the volume typed, and applying the rules again replaces them rather
  than adding a second copy;
* colour by BVS, CN and phi gives the colours the crystal path gives
  (facet.ui.md_scene with the theme's colour mode, theme.color_for_value);
* Qn and the modifier counts follow the bonds at v_bond, and are absent
  without formers (the descriptors that need them are greyed, not wrong);
* the channel rules read facet.core.md_channels (by charge, by voids, by
  modifier density) and the page says what it drew with which values;
* no verdict word appears on the page.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from conftest import dispose  # noqa: E402

HERE = Path(__file__).resolve().parent
QUARTZ = HERE / "data" / "crystals" / "quartz_SiO2_cod9013321.cif"
NS3 = Path(r"C:\Users\samso\AppData\Local\Temp\claude\C--Users-samso"
           r"\342b7e11-8171-41a0-8eee-4126254c9cc5\scratchpad"
           r"\facet_md_descriptors\models\ns3_pedone\nvt300K.lammpstrj")
NS3_TYPES = {1: "Si", 2: "O", 3: "Na"}
V_BOND, V_LIST = 0.075, 0.02

VERDICT_WORDS = ("good", "bad", "poor", "excellent", "acceptable", "correct",
                 "incorrect", "wrong", "reliable", "unreliable", "trustworthy",
                 "untrustworthy", "unusable", "should", "proves", "confirms")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _frame_data(frame, formers=("Si",), k=0):
    from facet.core import bulk, bv, md_model
    from facet.ui.md_highlight import FrameData

    ox = md_model.model_oxidation(frame.species)
    ox_atom = ox.per_atom(frame.elements)
    table, _ = bulk.analyse_frame(frame, ox_atom, bv.DEFAULT,
                                  v_bond_vu=V_BOND, v_list_vu=V_LIST)
    return FrameData.from_frame(frame, table, k=k, v_bond=V_BOND,
                                formers=formers, ox_atom=ox_atom)


@pytest.fixture(scope="module")
def quartz():
    from facet.core import md_model, readers

    frame, _ = md_model.supercell_frame(readers.read(QUARTZ), (3, 3, 3))
    return _frame_data(frame)


@pytest.fixture(scope="module")
def ns3():
    if not NS3.exists():
        pytest.skip(f"{NS3} is not on this machine")
    from facet.ui import md_jobs

    opened = md_jobs.open_model(str(NS3), {"type_map": NS3_TYPES})
    assert isinstance(opened, md_jobs.Opened), opened
    view = md_jobs.load_frame_view(opened.trajectory, 0, {}, None, V_BOND,
                                   V_LIST, with_bonds=True)
    from facet.ui.md_highlight import FrameData

    return FrameData.from_view(view, formers={"Si"})


def _scene(fd, theme=None):
    from facet.ui.md_scene import build_md_scene

    return build_md_scene(fd.frame, fd.table, theme=theme, v_bond=V_BOND,
                          v_list=V_LIST)


def _texts(widget) -> list[str]:
    from PySide6.QtWidgets import (QAbstractButton, QComboBox, QLabel,
                                   QLineEdit, QWidget)

    out = []
    for w in [widget] + widget.findChildren(QWidget):
        out.append(w.toolTip())
        if isinstance(w, QLabel):
            out.append(w.text())
        elif isinstance(w, QAbstractButton):
            out.append(w.text())
        elif isinstance(w, QLineEdit):
            out.append(w.placeholderText())
        elif isinstance(w, QComboBox):
            out.extend(w.itemText(i) for i in range(w.count()))
    return out


# ---------------------------------------------------------------------------
# the frame data
# ---------------------------------------------------------------------------

def test_frame_data_follows_the_bonds_at_v_bond(quartz):
    from facet.core import bulk
    from facet.ui.md_highlight import FrameData

    fd = quartz
    assert fd.n_atoms == 243 and set(fd.species) == {"Si", "O"}
    assert fd.formers == frozenset({"Si"})
    results = bulk.at_threshold(fd.table, V_BOND)
    assert np.array_equal(fd.cn, results.cn)
    assert np.array_equal(fd.bvs, results.bvs_vu)
    si, o = fd.elements == "Si", fd.elements == "O"
    # quartz: every O bridges two Si, every Si is Q4
    assert np.all(fd.former_cn[o] == 2) and np.all(fd.former_cn[si] == 0)
    assert np.all(fd.qn[si] == 4) and np.all(np.isnan(fd.qn[o]))
    assert np.all(fd.modifier_count[o] == 0)
    assert np.all(np.isnan(fd.modifier_count[si]))
    assert fd.cations == ("Si",) and fd.anions == ("O",)
    assert fd.modifiers == ("Si",)          # no modifier: every cation
    # without formers the former-dependent descriptors are absent
    bare = FrameData.from_frame(fd.frame, fd.table, v_bond=V_BOND)
    assert bare.qn is None and bare.former_cn is None \
        and bare.modifier_count is None
    with pytest.raises(ValueError, match="formers"):
        bare.descriptor("former CN")
    with pytest.raises(ValueError, match="formers"):
        bare.descriptor("Qn")
    assert np.array_equal(bare.descriptor("CN"), fd.cn)
    # a result-like object names the formers
    class _Glass:
        formers = frozenset({"Si"})

    assert FrameData.from_frame(fd.frame, fd.table, v_bond=V_BOND,
                                glass=_Glass()).formers == frozenset({"Si"})
    # the spheres are computed once, on demand
    assert fd._spheres is None
    spheres = fd.spheres
    assert len(spheres) > 0 and fd.spheres is spheres
    assert fd.sphere_centres_ang().shape == (len(spheres), 3)


# ---------------------------------------------------------------------------
# the rules on a scene
# ---------------------------------------------------------------------------

def test_show_only_keeps_what_a_direct_count_keeps(quartz):
    from facet.ui.md_highlight import Rule, apply_rules

    fd = quartz
    # Si with CN >= 4: every Si (81); a direct count from the results
    rule = Rule("show-only", element="Si", descriptor="CN", sense="≥",
                value=4)
    expected = int(((fd.elements == "Si") & (fd.cn >= 4)).sum())
    assert expected == 81
    scene = _scene(fd)
    before = scene.atom_radius.copy()
    before_colour = scene.atom_color.copy()
    n_before = scene.n_atoms
    result = apply_rules(scene, fd, [rule], dim_rest=True)
    assert result.n_matching == expected
    assert "81 of 243 atoms match (Si, CN ≥ 4)" in result.note
    assert "the rest dimmed" in result.note
    assert f"frame {fd.k} at v_bond {V_BOND:g} v.u." in result.note
    # dimmed: images included, every atom of a kept row keeps its radius,
    # every other atom has 40 % of it and a paler colour
    keep = ((fd.elements == "Si") & (fd.cn >= 4))[scene.atom_site]
    assert scene.n_atoms == n_before
    assert np.allclose(scene.atom_radius[keep], before[keep])
    assert np.allclose(scene.atom_radius[~keep], 0.4 * before[~keep])
    assert np.allclose(scene.atom_color[keep], before_colour[keep])
    assert np.any(scene.atom_color[~keep] != before_colour[~keep],
                  axis=1).all()
    # bonds to a kept atom keep their radius, the others shrink
    ends = scene.bond_atoms
    bond_keep = ((fd.elements == "Si") & (fd.cn >= 4))[ends[:, 0]] | \
        ((fd.elements == "Si") & (fd.cn >= 4))[ends[:, 1]]
    assert bond_keep.all()           # every bond has a Si end in quartz
    # hidden instead: only the kept rows (and their images) remain
    scene = _scene(fd)
    result = apply_rules(scene, fd, [rule], dim_rest=False)
    assert result.n_matching == expected
    assert "the rest left out" in result.note
    assert scene.n_atoms == int(keep.sum())
    assert set(scene.atom_element) == {"Si"}
    assert scene.n_bonds == 0           # no Si-Si bond survives
    assert len(scene.atom_label) == scene.n_atoms
    assert len(scene.atom_index) == scene.n_atoms
    # O with former CN <= 1 (non-bridging): none in quartz
    nbo = Rule("show-only", element="O", descriptor="former CN", sense="≤",
               value=1)
    scene = _scene(fd)
    result = apply_rules(scene, fd, [nbo], dim_rest=True)
    assert result.n_matching == 0
    # a disabled rule changes nothing
    scene = _scene(fd)
    reference = _scene(fd)
    off = Rule("show-only", enabled=False, element="O", descriptor="CN",
               sense="≥", value=9)
    result = apply_rules(scene, fd, [off], dim_rest=True)
    assert result.n_matching == fd.n_atoms
    assert np.allclose(scene.atom_radius, reference.atom_radius)
    assert np.allclose(scene.atom_color, reference.atom_color)


def test_void_spheres_are_drawn_once_at_the_volume_typed(quartz):
    from facet.ui.md_highlight import (Rule, apply_rules, radius_for_volume,
                                       sphere_mesh)

    fd = quartz
    radii = np.asarray(fd.spheres.radius_ang)
    # quartz is dense (largest empty sphere 0.36 Å): the volume typed is
    # one that keeps the largest third of the spheres
    r_70 = float(np.percentile(radii[radii > 0], 70))
    volume = round(4 / 3 * np.pi * r_70 ** 3, 4)
    r_min = radius_for_volume(volume)
    assert abs(4 / 3 * np.pi * r_min ** 3 - volume) < 1e-9
    expected = int((radii >= r_min).sum())
    assert 0 < expected < len(radii)
    rule = Rule("voids", min_volume_ang3=volume)
    scene = _scene(fd)
    result = apply_rules(scene, fd, [rule], dim_rest=True)
    assert result.n_spheres == expected
    voids = [m for m in scene.overlay_meshes if m.label == "voids"]
    assert len(voids) == 1
    assert voids[0].n_parts == expected
    assert voids[0].n_triangles == 80 * expected
    assert len(voids[0].part_of) == voids[0].n_triangles
    assert f"{expected} empty spheres ≥ {volume:g} Å³ (radius ≥ " \
        f"{r_min:.2f} Å)" in result.note
    # the sphere centres are the empty-sphere centres, in the scene's frame
    chosen = radii >= r_min
    assert np.allclose(voids[0].anchors, fd.sphere_centres_ang()[chosen],
                       atol=1e-4)
    # again on the same scene: replaced, not doubled
    result = apply_rules(scene, fd, [rule], dim_rest=True)
    assert len([m for m in scene.overlay_meshes if m.label == "voids"]) == 1
    assert result.n_spheres == expected
    # the mesh of one unit sphere has unit normals pointing outward
    mesh = sphere_mesh([[0, 0, 0]], [2.0])
    assert np.allclose(np.linalg.norm(mesh.vertices, axis=1), 2.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-5)
    assert np.all((mesh.vertices * mesh.normals).sum(axis=1) > 0)
    # a volume above every sphere draws nothing, and says so
    scene = _scene(fd)
    result = apply_rules(scene, fd, [Rule("voids", min_volume_ang3=1e6)],
                         dim_rest=True)
    assert result.n_spheres == 0
    assert not [m for m in scene.overlay_meshes if m.label == "voids"]
    assert "0 empty spheres ≥ 1e+06 Å³" in result.note


def test_colour_by_bvs_cn_and_phi_match_the_crystal_path(quartz):
    from facet.core import theme as theme_mod
    from facet.ui.md_highlight import apply_rules

    fd = quartz
    for mode, name in ((theme_mod.ColorMode.BVS, "bond-valence sum"),
                       (theme_mod.ColorMode.COORDINATION,
                        "coordination number"),
                       (theme_mod.ColorMode.PHI, "stereoactivity (phi)")):
        theme = theme_mod.Theme()
        theme.color_mode = mode
        reference = _scene(fd, theme)       # md_scene's own colouring
        plain = theme_mod.Theme()
        scene = _scene(fd, plain)
        result = apply_rules(scene, fd, [], dim_rest=True, colour_by=name,
                             theme=plain)
        assert np.allclose(scene.atom_color, reference.atom_color), name
        assert f"colour: {name}" in result.note
        assert not result.problems
    # element colours come back on a scene built in another mode
    theme = theme_mod.Theme()
    theme.color_mode = theme_mod.ColorMode.BVS
    scene = _scene(fd, theme)
    apply_rules(scene, fd, [], dim_rest=True, colour_by="element",
                theme=theme_mod.Theme())
    assert np.allclose(scene.atom_color, _scene(fd).atom_color)
    # a direct check of one cation against color_for_value
    cations = np.nonzero(~np.asarray(fd.table.is_anion))[0]
    values = {int(i): float(fd.bvs[i]) for i in cations}
    lo, hi = theme_mod.scale_range(values, theme_mod.Theme(),
                                   theme_mod.ColorMode.BVS)
    scene = _scene(fd)
    apply_rules(scene, fd, [], dim_rest=True, colour_by="bond-valence sum")
    row = int(cations[0])
    where = np.nonzero(np.asarray(scene.atom_site) == row)[0][0]
    assert np.allclose(scene.atom_color[where],
                       theme_mod.color_for_value(values[row], lo, hi),
                       atol=1e-6)


def test_qn_and_modifier_colours_use_their_own_ramps(quartz):
    from facet.core import theme as theme_mod
    from facet.ui.md_highlight import FrameData, apply_rules

    fd = quartz
    scene = _scene(fd)
    result = apply_rules(scene, fd, [], dim_rest=True, colour_by="Qn")
    assert not result.problems
    si = np.asarray(scene.atom_element) == "Si"
    top = theme_mod.color_for_value(4.0, 0.0, 4.0)
    assert np.allclose(scene.atom_color[si], top, atol=1e-6)
    element = _scene(fd).atom_color
    assert np.allclose(scene.atom_color[~si], element[~si])     # O: element
    scene = _scene(fd)
    result = apply_rules(scene, fd, [], dim_rest=True,
                         colour_by="modifier-rich O")
    o = np.asarray(scene.atom_element) == "O"
    assert np.allclose(scene.atom_color[o],
                       theme_mod.color_for_value(0.0, 0.0, 1.0), atol=1e-6)
    # without formers: a problem is reported and element colours stay
    bare = FrameData.from_frame(fd.frame, fd.table, v_bond=V_BOND)
    scene = _scene(bare)
    result = apply_rules(scene, bare, [], dim_rest=True, colour_by="Qn")
    assert result.problems and "formers" in result.problems[0]
    assert np.allclose(scene.atom_color, element)
    # channel membership without a channel rule: reported, not raised
    scene = _scene(fd)
    result = apply_rules(scene, fd, [], dim_rest=True,
                         colour_by="channel membership")
    assert any("channels rule" in p for p in result.problems)


def test_channels_by_charge_draw_the_regions_of_a_probe(quartz):
    from facet.ui.md_highlight import Rule, apply_rules, channel_backend

    ok, reason = channel_backend("by charge")
    assert ok, reason
    fd = quartz
    rule = Rule("channels", channel_by="by charge", threshold=0.5, probe="Na",
                grid_spacing_ang=1.5, r_cut_ang=4.0)
    scene = _scene(fd)
    result = apply_rules(scene, fd, [rule], dim_rest=True,
                         colour_by="channel membership")
    assert not result.problems, result.problems
    assert "channels by charge: Na1+ probe, mismatch ≤ 0.5 v.u. on a 1.5 Å " \
        "grid (r_cut 4 Å)" in result.note
    assert "colour: channel membership" in result.note
    landscape = fd.landscape("Na", 1.5, 4.0)
    assert landscape.probe == "Na" and landscape.probe_ox == 1
    assert fd.landscape("Na", 1.5, 4.0) is landscape        # kept
    from facet.core import md_channels

    regions = md_channels.accessible_regions(landscape, 0.5)
    assert result.n_regions == regions.n_regions
    inside = np.asarray(regions.region_of_atom) >= 0
    assert result.n_in_regions == int(inside.sum())
    channels = [m for m in scene.overlay_meshes if m.label == "channels"]
    if regions.n_regions:
        assert len(channels) == 1 and channels[0].n_triangles > 0
        # the atoms outside every region are dimmed
        keep = inside[np.asarray(scene.atom_site)]
        reference = _scene(fd)
        assert np.allclose(scene.atom_radius[~keep],
                           0.4 * reference.atom_radius[~keep])
    # a rule without a grid is refused in the note, never raised
    scene = _scene(fd)
    result = apply_rules(scene, fd, [Rule("channels", channel_by="by charge",
                                          probe="Na")], dim_rest=True)
    assert result.problems and "grid spacing" in result.problems[0]
    assert "not drawn" in result.note


def test_void_regions_filter_by_elongation_when_the_build_has_them(quartz):
    from facet.ui.md_highlight import Rule, apply_rules, channel_backend

    ok, reason = channel_backend("by voids")
    fd = quartz
    radii = np.asarray(fd.spheres.radius_ang)
    volume = round(4 / 3 * np.pi * float(np.percentile(radii[radii > 0],
                                                        50)) ** 3, 4)
    scene = _scene(fd)
    plain = apply_rules(scene, fd, [Rule("voids", min_volume_ang3=volume)],
                        dim_rest=True)
    scene = _scene(fd)
    elongated = apply_rules(scene, fd, [Rule("voids", min_volume_ang3=volume,
                                             min_elongation=1.2,
                                             grid_spacing_ang=0.5)],
                            dim_rest=True)
    if not ok:
        assert elongated.problems and reason in elongated.problems[0]
        assert elongated.n_spheres == plain.n_spheres
        return
    assert not elongated.problems, elongated.problems
    assert elongated.n_spheres <= plain.n_spheres
    assert "elongation ≥ 1.2" in elongated.note
    regions = fd.void_regions(0.0, 0.5)
    assert fd.void_regions(0.0, 0.5) is regions
    keep = np.asarray(regions.elongation) >= 1.2
    rows = np.asarray(regions.sphere_rows)[
        keep[np.asarray(regions.sphere_region)]]
    from facet.ui.md_highlight import radius_for_volume

    expected = int((radii[rows] >= radius_for_volume(volume)).sum())
    assert elongated.n_spheres == expected
    # the channel rule by voids
    scene = _scene(fd)
    result = apply_rules(scene, fd, [Rule("channels", channel_by="by voids",
                                          threshold=1.2, r_cut_ang=0.0,
                                          grid_spacing_ang=0.5)],
                         dim_rest=True, colour_by="channel membership")
    assert not result.problems, result.problems
    assert "channels by voids" in result.note
    assert result.n_regions == int(keep.sum())


def test_channels_by_modifier_density_on_the_glass_model(ns3):
    from facet.ui.md_highlight import Rule, apply_rules, channel_backend

    ok, reason = channel_backend("by modifier density")
    assert ok, reason
    fd = ns3
    assert fd.modifiers == ("Na",)
    rule = Rule("channels", channel_by="by modifier density", threshold=2,
                probe="Na", r_cut_ang=3.0)
    scene = _scene(fd)
    result = apply_rules(scene, fd, [rule], dim_rest=True,
                         colour_by="channel membership")
    assert not result.problems, result.problems
    assert "anions with ≥ 2 Na within 3 Å" in result.note
    assert result.n_regions >= 1 and result.n_in_regions > 0
    # the non-bridging oxygens of the glass, counted two ways
    nbo = Rule("show-only", element="O", descriptor="former CN", sense="≤",
               value=1)
    scene = _scene(fd)
    result = apply_rules(scene, fd, [nbo], dim_rest=True)
    expected = int(((fd.elements == "O") & (fd.former_cn <= 1)).sum())
    assert result.n_matching == expected
    assert "atoms match (O, former CN ≤ 1)" in result.note


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------

def test_the_page_offers_the_frame_s_species_and_applies_its_rules(qapp,
                                                                     quartz):
    from facet.ui.md_highlight import (COLOUR_MODES, DESCRIPTORS, FrameData,
                                       HighlightPage)

    page = HighlightPage()
    try:
        assert [r.kind for r in page.rules()] == ["show-only", "voids",
                                                  "channels"]
        assert not any(r.enabled for r in page.rules())
        assert page.apply_to(_scene(quartz)) is None
        page.set_frame_data(quartz)
        assert [page.only.element.itemText(i)
                for i in range(page.only.element.count())] == \
            ["any"] + list(quartz.species)
        assert [page.channels.probe.itemText(i)
                for i in range(page.channels.probe.count())] == \
            list(quartz.modifiers)
        assert [page.colour.itemText(i)
                for i in range(page.colour.count())] == list(COLOUR_MODES)
        assert [page.only.descriptor.itemText(i)
                for i in range(page.only.descriptor.count())] == \
            list(DESCRIPTORS)
        changes = []
        page.rulesChanged.connect(lambda: changes.append(True))
        page.only.tick.setChecked(True)
        page.only.element.setCurrentText("Si")
        page.only.descriptor.setCurrentText("CN")
        page.only.sense.setCurrentText("≥")
        page.only.value.setValue(4)
        page.voids.tick.setChecked(True)
        page.voids.volume.setValue(4.0)
        assert len(changes) >= 5
        rules = page.rules()
        assert rules[0].enabled and rules[0].element == "Si" \
            and rules[0].sense == "≥" and rules[0].value == 4
        assert rules[1].enabled and rules[1].min_volume_ang3 == 4.0
        assert not rules[2].enabled
        scene = _scene(quartz)
        result = page.apply_to(scene)
        assert result.n_matching == 81
        assert page.summary.text() == result.note
        assert page.result is result
        assert "the rest dimmed" in result.note
        page.dim.setChecked(False)
        scene = _scene(quartz)
        result = page.apply_to(scene)
        assert "the rest left out" in result.note and scene.n_atoms < 243
        # BVS and phi take decimals, CN and Qn do not
        page.only.descriptor.setCurrentText("BVS")
        assert page.only.value.decimals() == 2
        page.only.descriptor.setCurrentText("CN")
        assert page.only.value.decimals() == 0
        # a rule added and removed
        row = page.add_rule("show-only")
        assert len(page.rules()) == 4
        assert row.element.count() == 1 + len(quartz.species)
        page.remove_rule(row)
        assert len(page.rules()) == 3
        # without formers the former-dependent descriptors are greyed
        bare = FrameData.from_frame(quartz.frame, quartz.table, v_bond=V_BOND)
        page.set_frame_data(bare)
        model = page.only.descriptor.model()
        assert not model.item(DESCRIPTORS.index("former CN")).isEnabled()
        assert not model.item(DESCRIPTORS.index("Qn")).isEnabled()
        assert "formers" in model.item(DESCRIPTORS.index("Qn")).toolTip()
        assert not page.colour.model().item(
            COLOUR_MODES.index("Qn")).isEnabled()
        page.set_frame_data(quartz)
        assert model.item(DESCRIPTORS.index("Qn")).isEnabled()
    finally:
        dispose(page)


def test_the_channel_row_follows_its_mode_and_the_build(qapp, quartz):
    from facet.ui.md_highlight import (CHANNEL_MODES, HighlightPage,
                                       channel_backend)

    page = HighlightPage()
    try:
        page.set_frame_data(quartz)
        row = page.channels
        model = row.mode.model()
        for i, mode in enumerate(CHANNEL_MODES):
            ok, reason = channel_backend(mode)
            assert model.item(i).isEnabled() == ok, mode
            if not ok:
                assert reason in model.item(i).toolTip()
        row.mode.setCurrentText("by charge")
        assert row.threshold_label.text() == "mismatch ≤"
        assert row.threshold.suffix() == " v.u."
        assert row.cut_label.text() == "r_cut"
        rule = row.rule()
        assert rule.channel_by == "by charge" and rule.probe == "Si"
        assert rule.grid_spacing_ang == 0.5 and rule.r_cut_ang == 6.0
        if channel_backend("by modifier density")[0]:
            row.mode.setCurrentText("by modifier density")
            assert row.threshold_label.text() == "modifiers ≥"
            assert row.threshold.decimals() == 0
            assert row.cut_label.text() == "M–anion cutoff"
            assert row.rule().threshold == 2
        if channel_backend("by voids")[0]:
            row.mode.setCurrentText("by voids")
            assert row.threshold_label.text() == "elongation ≥"
            assert row.cut_label.text() == "probe radius"
            assert row.probe.isHidden() or not row.probe.isVisibleTo(row)
        voids_ok, voids_reason = channel_backend("by voids")
        assert page.voids.elongation.isEnabled() == voids_ok
        if not voids_ok:
            assert voids_reason in page.voids.note.text()
            assert page.voids.rule().min_elongation is None
    finally:
        dispose(page)


def test_no_verdict_word_appears_on_the_page_and_it_fits_the_column(qapp,
                                                                     quartz):
    from facet.ui import chrome
    from facet.ui.md_highlight import HighlightPage

    pattern = re.compile(r"\b(" + "|".join(VERDICT_WORDS) + r")\b",
                         re.IGNORECASE)
    page = HighlightPage()
    area = chrome.in_scroll_area(page)
    try:
        page.set_frame_data(quartz)
        page.only.tick.setChecked(True)
        page.voids.tick.setChecked(True)
        page.apply_to(_scene(quartz))
        found = {}
        for text in _texts(page):
            for match in pattern.finditer(text or ""):
                found.setdefault(match.group(0).lower(), text[:160])
        assert not found, found
        area.resize(470, 700)
        area.show()
        qapp.processEvents()
        assert page.minimumSizeHint().width() <= 470
        assert page.width() <= 470
        for row in page.rows:
            assert row.width() <= 470 - 16
    finally:
        dispose(area)


# ---------------------------------------------------------------------------
# cancellation and the grid cap on the page
# ---------------------------------------------------------------------------

def test_frame_data_forwards_the_cancel_hook(quartz):
    import time

    from facet.core import md_channels

    clock = time.perf_counter()
    with pytest.raises(md_channels.AnalysisCancelled):
        quartz.landscape("Si", 0.4, 5.0, cancelled=lambda: True)
    assert time.perf_counter() - clock < 1.0
    assert ("Si", 0.4, 5.0) not in quartz._landscapes
    with pytest.raises(md_channels.AnalysisCancelled):
        quartz.void_regions(0.0, 0.45, cancelled=lambda: True)
    assert (0.0, 0.45) not in quartz._void_regions


def test_the_rows_state_the_grid_before_anything_runs(qapp, quartz,
                                                      monkeypatch):
    from facet.core import md_channels
    from facet.ui.md_highlight import HighlightPage, grid_points_note

    page = HighlightPage()
    try:
        page.set_frame_data(quartz)
        row = page.channels
        row.mode.setCurrentText("by charge")
        row.spacing.setValue(0.5)
        shape, n_points, _ = md_channels.grid_guard(quartz.frame.box_ang, 0.5)
        note = row.note.text()
        assert f"{shape[0]} × {shape[1]} × {shape[2]}" in note
        assert "0.5" in note
        # the voids row states its own 0.5 Å union grid
        assert grid_points_note(quartz, 0.5) in page.voids.note.text()
        # a grid the cap refuses shows the engine's refusal, before any run
        monkeypatch.setattr(md_channels, "MAX_GRID_POINTS", 10)
        row.spacing.setValue(0.4)
        note = row.note.text()
        assert "more than the 10" in note and "would fit" in note
        assert grid_points_note(quartz, 0.4) == note
    finally:
        dispose(page)


def test_a_live_highlight_job_is_cancelled_at_interpreter_exit():
    """The crash this pins: a highlight worker computing a large landscape,
    the window closed with the job running, the interpreter tearing down a
    live QThread (STATUS_STACK_BUFFER_OVERRUN). A child process starts a
    highlight-style Job on a deliberately slow landscape (one grid point a
    chunk), prints that it is still running, and ends: the keeper's atexit
    wait cancels the job, the engine's per-chunk poll raises, and the
    process exits 0 well inside the deadline."""
    import os
    import subprocess
    import sys
    import time

    code = """
import time
import numpy as np
from PySide6.QtWidgets import QApplication
from facet.core import bulk, bv, md_model
from facet.core import md_channels
from facet.core.md_model import frame_from_arrays
from facet.ui import md_jobs
from facet.ui.md_highlight import FrameData

md_channels.PAIRS_PER_CHUNK = 1   # one grid point a chunk: slow, polled per chunk
app = QApplication([])
rng = np.random.default_rng(3)
frame = frame_from_arrays(["Na"] * 10 + ["O"] * 30,
                          cart_ang=rng.uniform(0.0, 12.0, (40, 3)),
                          box_ang=np.eye(3) * 12.0)
ox_atom = md_model.model_oxidation(frame.species).per_atom(frame.elements)
table, _ = bulk.analyse_frame(frame, ox_atom, bv.DEFAULT, v_bond_vu=0.075,
                              v_list_vu=0.02)
fd = FrameData.from_frame(frame, table, v_bond=0.075, ox_atom=ox_atom)
job = md_jobs.Job(lambda progress, cancelled: fd.landscape("Na", 0.1, 5.0),
                  name="highlight")
job.start()
time.sleep(1.5)
print("RUNNING" if job.is_running() else "ENDED", flush=True)
# the script ends with the worker thread alive: the keeper cancels and
# joins it at interpreter exit
"""
    clock = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=str(HERE.parent),
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True, text=True, timeout=120)
    elapsed = time.perf_counter() - clock
    assert result.returncode == 0, (result.returncode, result.stderr[-2000:])
    assert "RUNNING" in result.stdout, result.stdout
    assert elapsed < 60, elapsed
