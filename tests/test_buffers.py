"""CPU geometry expansion.

A wrong basis or a transposed axis here looks like a rendering bug and is very
hard to find by eye, so the geometry is checked numerically instead.
"""
from __future__ import annotations

import numpy as np
import pytest

from facet.gl import buffers
from facet.gl.scene import Scene


# --- picking ids -------------------------------------------------------------

@pytest.mark.parametrize("i", [0, 1, 254, 255, 256, 65535, 65536, 1_000_000])
def test_object_ids_round_trip_through_a_colour(i):
    rgb = buffers.id_to_rgb(np.array([i]))[0]
    px = [int(round(c * 255)) for c in rgb]
    assert buffers.rgb_to_id(*px) == i


def test_the_background_decodes_to_nothing():
    assert buffers.rgb_to_id(0, 0, 0) is None


def test_ids_are_distinct_across_a_large_scene():
    ids = buffers.id_to_rgb(np.arange(20000))
    px = np.round(ids * 255).astype(int)
    assert len({tuple(p) for p in px}) == 20000


# --- spheres -----------------------------------------------------------------

def _scene_with_atoms(n=4):
    s = Scene()
    rng = np.random.default_rng(0)
    s.atom_position = rng.normal(size=(n, 3)).astype(np.float32)
    s.atom_radius = np.full(n, 0.5, np.float32)
    s.atom_color = np.tile(np.array([0.5, 0.2, 0.1], np.float32), (n, 1))
    s.atom_index = np.arange(n, dtype=np.int32)
    return s


def test_every_atom_becomes_six_vertices():
    s = _scene_with_atoms(7)
    v = buffers.sphere_vertices(s)
    assert len(v["aCorner"]) == 7 * buffers.VERTS_PER_SPHERE
    for arr in v.values():
        assert len(arr) == 7 * buffers.VERTS_PER_SPHERE


def test_all_six_vertices_of_an_atom_share_its_centre():
    s = _scene_with_atoms(3)
    v = buffers.sphere_vertices(s)
    for i in range(3):
        block = v["aCenter"][i * 6:(i + 1) * 6]
        assert np.allclose(block, s.atom_position[i])


def test_the_quad_covers_the_sphere_in_both_directions():
    v = buffers.sphere_vertices(_scene_with_atoms(1))
    c = v["aCorner"]
    assert c[:, 0].min() == -1 and c[:, 0].max() == 1
    assert c[:, 1].min() == -1 and c[:, 1].max() == 1


def test_empty_scene_produces_empty_arrays_not_an_error():
    v = buffers.sphere_vertices(Scene())
    assert all(len(a) == 0 for a in v.values())
    t = buffers.tube_vertices(Scene())
    assert all(len(a) == 0 for a in t.values())


# --- tubes -------------------------------------------------------------------

def _scene_with_bond(a, b, radius=0.1):
    s = Scene()
    s.bond_a = np.array([a], np.float32)
    s.bond_b = np.array([b], np.float32)
    s.bond_radius = np.array([radius], np.float32)
    s.bond_color_a = np.array([[1.0, 0.0, 0.0]], np.float32)
    s.bond_color_b = np.array([[0.0, 0.0, 1.0]], np.float32)
    s.bond_valence = np.array([0.5], np.float32)
    return s


def test_unit_tube_is_split_into_two_equal_halves():
    pos, nrm, param = buffers.unit_tube(12)
    assert set(np.unique(param)) == {0.0, 1.0}
    assert (param == 0.0).sum() == (param == 1.0).sum()


def test_unit_tube_normals_are_radial_and_unit_length():
    pos, nrm, _ = buffers.unit_tube(12)
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-6)
    assert np.allclose(nrm[:, 2], 0.0, atol=1e-7)


def test_tube_spans_exactly_between_its_two_atoms():
    a, b = (1.0, 2.0, 3.0), (1.0, 2.0, 5.0)
    v = buffers.tube_vertices(_scene_with_bond(a, b), sides=12)
    z = v["aPosition"][:, 2]
    assert z.min() == pytest.approx(3.0, abs=1e-5)
    assert z.max() == pytest.approx(5.0, abs=1e-5)


def test_tube_radius_is_respected_perpendicular_to_the_axis():
    a, b = (0.0, 0.0, 0.0), (0.0, 0.0, 4.0)
    v = buffers.tube_vertices(_scene_with_bond(a, b, radius=0.25), sides=24)
    radial = np.linalg.norm(v["aPosition"][:, :2], axis=1)
    assert np.allclose(radial, 0.25, atol=1e-6)


def test_tube_normals_are_perpendicular_to_the_bond_axis():
    a, b = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    v = buffers.tube_vertices(_scene_with_bond(a, b), sides=12)
    axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    dots = v["aNormal"] @ axis
    assert np.allclose(dots, 0.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(v["aNormal"], axis=1), 1.0, atol=1e-5)


def test_the_first_half_carries_the_first_atoms_colour():
    a, b = (0.0, 0.0, 0.0), (0.0, 0.0, 2.0)
    v = buffers.tube_vertices(_scene_with_bond(a, b), sides=12)
    near = v["aParam"][:, 0] < 0.5
    assert np.allclose(v["aColor"][near], [1.0, 0.0, 0.0])
    assert np.allclose(v["aColorB"][~near], [0.0, 0.0, 1.0])
    # the half nearer atom A must actually be nearer atom A
    assert v["aPosition"][near][:, 2].max() <= 1.0 + 1e-5


@pytest.mark.parametrize("direction", [
    (0, 0, 1), (0, 0, -1), (1, 0, 0), (0, 1, 0), (1, 1, 1), (-3, 0.001, 0.002),
])
def test_basis_never_degenerates_whatever_the_bond_direction(direction):
    """A single fixed helper vector collapses for some bond in almost any
    structure; the basis must stay finite and orthonormal for every direction."""
    d = np.array(direction, float)
    d = d / np.linalg.norm(d)
    v = buffers.tube_vertices(_scene_with_bond((0, 0, 0), tuple(d * 2.0)), sides=8)
    assert np.all(np.isfinite(v["aPosition"]))
    assert np.all(np.isfinite(v["aNormal"]))
    assert np.allclose(np.linalg.norm(v["aNormal"], axis=1), 1.0, atol=1e-5)
    assert np.abs(v["aNormal"] @ d).max() < 1e-5


def test_many_bonds_expand_independently():
    s = Scene()
    rng = np.random.default_rng(2)
    m = 25
    s.bond_a = rng.normal(size=(m, 3)).astype(np.float32)
    s.bond_b = (s.bond_a + rng.normal(size=(m, 3))).astype(np.float32)
    s.bond_radius = np.full(m, 0.08, np.float32)
    s.bond_color_a = np.zeros((m, 3), np.float32)
    s.bond_color_b = np.ones((m, 3), np.float32)
    v = buffers.tube_vertices(s, sides=10)
    assert len(v["aPosition"]) == m * 10 * 12
    assert np.all(np.isfinite(v["aPosition"]))


# --- lines and budget --------------------------------------------------------

def test_line_segments_flatten_to_pairs():
    seg = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    out = buffers.line_vertices(seg)
    assert out.shape == (4, 3)
    assert np.allclose(out[0], seg[0, 0]) and np.allclose(out[3], seg[1, 1])


def test_tube_resolution_drops_for_large_scenes_but_stays_usable():
    small, big = Scene(), Scene()
    small.bond_a = np.zeros((10, 3), np.float32)
    big.bond_a = np.zeros((10, 3), np.float32)
    small.bond_radius = np.zeros(10, np.float32)
    big.bond_radius = np.zeros(20000, np.float32)
    assert buffers.tube_sides_for(small) == 20
    reduced = buffers.tube_sides_for(big)
    assert reduced < 20 and reduced >= 5
