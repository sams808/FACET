"""Label text and placement.

Placement takes a text-measuring callable rather than a painter, so the
de-cluttering can be checked without a window.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from facet.gl import labels as L

SAMPLE = Path(r"C:\Users\samso\Desktop\WSU_work\XRD\cif\Bi\cifs\1526458_Bi2O3.cif")


def _measure(text: str) -> tuple[float, float]:
    """A stub metric: 7 px per character, 14 px tall."""
    return (7.0 * len(text), 14.0)


# --- text --------------------------------------------------------------------

def test_atom_text_covers_every_kind_without_raising():
    class Site:
        ox = 3
        wyckoff = "e"
        multiplicity = 4

    class Result:
        cn_valence = 5
        bvs = 2.93
        phi = 0.433

    for kind in L.AtomLabel:
        text = L.atom_text(kind, element="Bi", label="Bi1", index=1,
                           occupancy=1.0, result=Result(), site=Site(),
                           frac=np.array([0.1, 0.2, 0.3]))
        assert isinstance(text, str)


@pytest.mark.parametrize("kind,expected", [
    (L.AtomLabel.NONE, ""),
    (L.AtomLabel.ELEMENT, "Bi"),
    (L.AtomLabel.SITE, "Bi1"),
    (L.AtomLabel.ELEMENT_INDEX, "Bi1"),
])
def test_simple_atom_labels(kind, expected):
    assert L.atom_text(kind, element="Bi", label="Bi1", index=1,
                       occupancy=1.0) == expected


def test_oxidation_label_uses_the_sign():
    class Site:
        ox = 3

    assert L.atom_text(L.AtomLabel.OXIDATION, element="Bi", label="Bi1",
                       index=1, occupancy=1.0, site=Site()) == "Bi3+"

    class Anion:
        ox = -2

    assert L.atom_text(L.AtomLabel.OXIDATION, element="O", label="O1",
                       index=1, occupancy=1.0, site=Anion()) == "O2-"


def test_a_fully_occupied_site_gets_no_occupancy_label():
    """A full site says nothing interesting; only partial occupancy does."""
    assert L.atom_text(L.AtomLabel.OCCUPANCY, element="Bi", label="Bi1",
                       index=1, occupancy=1.0) == ""
    assert L.atom_text(L.AtomLabel.OCCUPANCY, element="Bi", label="Bi1",
                       index=1, occupancy=0.92) == "0.92"


def test_analysis_labels_need_a_result():
    assert L.atom_text(L.AtomLabel.BVS, element="Bi", label="Bi1",
                       index=1, occupancy=1.0, result=None) == ""


@pytest.mark.parametrize("kind,expected", [
    (L.BondLabel.NONE, ""),
    (L.BondLabel.DISTANCE, "2.119 \u00c5"),
    (L.BondLabel.VALENCE, "0.924 v.u."),
    (L.BondLabel.BOTH, "2.119 \u00c5 \u00b7 0.924 v.u."),
])
def test_bond_label_text(kind, expected):
    assert L.bond_text(kind, distance=2.1194, valence=0.9237) == expected


def test_percent_of_shortest_needs_a_reference():
    assert L.bond_text(L.BondLabel.PERCENT, distance=2.5, valence=0.3) == ""
    assert L.bond_text(L.BondLabel.PERCENT, distance=2.5, valence=0.3,
                       shortest=2.0) == "125 %"


def test_share_of_the_sum():
    assert L.bond_text(L.BondLabel.FRACTION_OF_BVS, distance=2.1,
                       valence=0.9, bvs=3.0) == "30 %"


# --- placement ---------------------------------------------------------------

@pytest.mark.skipif(not SAMPLE.exists(), reason="sample structure not present")
class TestPlacement:

    @pytest.fixture(scope="class")
    def context(self):
        from facet.core import cif, coordination
        from facet.gl.camera import Camera
        from facet.gl.scene import build_scene

        s = cif.read(SAMPLE)
        res = coordination.analyse_structure(s)
        scene = build_scene(s, res)
        cam = Camera()
        cam.frame(scene.center, scene.radius)
        return s, res, scene, cam

    def _build(self, context, settings, **kw):
        s, res, scene, cam = context
        return L.build(scene, cam, settings, width=900, height=700,
                       measure=_measure, results=res, structure=s, **kw)

    def test_nothing_enabled_produces_nothing(self, context):
        assert self._build(context, L.LabelSettings()) == []

    def test_atom_labels_are_produced(self, context):
        out = self._build(context, L.LabelSettings(atom=L.AtomLabel.SITE))
        assert out and all(p.kind == "atom" for p in out)

    def test_decluttering_drops_overlapping_labels(self, context):
        crowded = L.LabelSettings(atom=L.AtomLabel.SITE, declutter=True)
        loose = L.LabelSettings(atom=L.AtomLabel.SITE, declutter=False)
        assert len(self._build(context, crowded)) < len(self._build(context, loose))

    def test_declutter_leaves_no_overlapping_boxes(self, context):
        out = self._build(context, L.LabelSettings(atom=L.AtomLabel.SITE))
        boxes = []
        for p in out:
            w, h = _measure(p.text)
            box = (p.x, p.y - h, p.x + w, p.y)
            for b in boxes:
                assert not (box[0] < b[2] and box[2] > b[0]
                            and box[1] < b[3] and box[3] > b[1]), \
                    f"{p.text} overlaps a placed label"
            boxes.append(box)

    def test_nearest_labels_win_a_collision(self, context):
        """When two collide the front one should survive, because it belongs
        to what the viewer is looking at."""
        out = self._build(context, L.LabelSettings(atom=L.AtomLabel.SITE))
        assert out[0].depth == max(p.depth for p in out)

    def test_cation_scope_excludes_anions(self, context):
        out = self._build(context, L.LabelSettings(
            atom=L.AtomLabel.ELEMENT, atom_scope=L.LabelScope.CATIONS))
        assert out and all(p.text == "Bi" for p in out)

    def test_anion_scope_excludes_cations(self, context):
        out = self._build(context, L.LabelSettings(
            atom=L.AtomLabel.ELEMENT, atom_scope=L.LabelScope.ANIONS))
        assert out and all(p.text == "O" for p in out)

    def test_selected_site_scope_narrows_to_one_site(self, context):
        s, res, scene, cam = context
        out = self._build(context, L.LabelSettings(
            atom=L.AtomLabel.SITE, atom_scope=L.LabelScope.SELECTED_SITE),
            selected_site=res[0].site_index)
        assert out and all(p.text == res[0].label for p in out)

    def test_bond_labels_default_to_bonds_only(self, context):
        s, res, scene, cam = context
        out = self._build(context, L.LabelSettings(bond=L.BondLabel.VALENCE))
        assert out
        for p in out:
            value = float(p.text.split()[0])
            assert value >= scene.v_bond - 1e-9

    def test_bond_scope_all_includes_sub_threshold_contacts(self, context):
        bonded = self._build(context, L.LabelSettings(bond=L.BondLabel.VALENCE))
        every = self._build(context, L.LabelSettings(
            bond=L.BondLabel.VALENCE, bond_scope=L.LabelScope.ALL,
            declutter=False))
        assert len(every) > len(bonded)

    def test_axis_labels_are_a_b_c(self, context):
        out = self._build(context, L.LabelSettings(show_axes=True))
        assert out
        assert {p.text for p in out} <= {"a", "b", "c"}

    def test_the_hard_cap_is_respected(self, context):
        out = self._build(context, L.LabelSettings(
            atom=L.AtomLabel.SITE, declutter=False, max_labels=5))
        assert len(out) == 5

    def test_labels_stay_near_the_viewport(self, context):
        out = self._build(context, L.LabelSettings(atom=L.AtomLabel.SITE))
        for p in out:
            assert -50 <= p.x <= 950
            assert 0 <= p.y <= 720

    def test_measurement_labels_give_a_distance_then_an_angle(self, context):
        s, res, scene, cam = context
        out = L.measurement_labels(scene, cam, [0, 1, 2], width=900, height=700)
        texts = [p.text for p in out]
        assert any("\u00c5" in t for t in texts)
        assert any("\u00b0" in t for t in texts)

    def test_a_measurement_chain_of_one_gives_nothing(self, context):
        s, res, scene, cam = context
        assert L.measurement_labels(scene, cam, [0], width=900, height=700) == []
