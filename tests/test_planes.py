"""Lattice planes and slabs.

The plane geometry is checked against closed forms -- a (001) section of a cubic
cell is a square of side a, a (111) section through a corner is an equilateral
triangle -- and against the definition: the normal of (hkl) must be
perpendicular to every lattice vector [uvw] with hu + kv + lw = 0. That last one
is the check that matters, because the tempting wrong answer (h a + k b + l c)
passes every cubic test and is more than fifty degrees out for hexagonal (111).

The slab is checked for the thing that actually breaks: it renumbers atoms, so
every array indexed by atom has to move with them. A bond that survives must
still point at the same two atoms it did before.
"""
from __future__ import annotations

import itertools
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from facet.core import cif
from facet.core import planes as P
from facet.core.utilities import d_spacing

HKL = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (1, 1, 1),
       (2, 1, 0), (1, -1, 2), (3, 1, -2), (0, 2, 1)]


def _build(a, b, c, alpha, beta, gamma, spacegroup, rows):
    header = ["data_test", f"_cell_length_a {a}", f"_cell_length_b {b}",
              f"_cell_length_c {c}", f"_cell_angle_alpha {alpha}",
              f"_cell_angle_beta {beta}", f"_cell_angle_gamma {gamma}",
              f"_symmetry_space_group_name_H-M '{spacegroup}'", "loop_",
              "_atom_site_label", "_atom_site_type_symbol",
              "_atom_site_fract_x", "_atom_site_fract_y",
              "_atom_site_fract_z", "_atom_site_occupancy"]
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "test.cif")
    Path(path).write_text("\n".join(header + list(rows)) + "\n",
                          encoding="utf-8")
    return cif.read(path)


SYSTEMS = {
    "cubic": (5.6402, 5.6402, 5.6402, 90, 90, 90, "F m -3 m",
              ["Na1 Na 0 0 0 1.0", "Cl1 Cl 0.5 0.5 0.5 1.0"]),
    "hexagonal": (5.44, 5.44, 12.9, 90, 90, 120, "P 6_3/m m c",
                  ["Bi1 Bi 0.3333 0.6667 0.25 1.0", "O1 O 0 0 0 1.0"]),
    "tetragonal": (3.79, 3.79, 9.51, 90, 90, 90, "I 4/m m m",
                   ["Bi1 Bi 0 0 0 1.0", "O1 O 0 0.5 0.25 1.0"]),
    "monoclinic": (5.85, 8.17, 7.51, 90, 113.0, 90, "P 1 21/c 1",
                   ["Bi1 Bi 0.02 0.04 0.24 1.0", "O1 O 0.78 0.31 0.21 1.0"]),
    "triclinic": (6.10, 7.20, 8.30, 88, 102, 95, "P -1",
                  ["Bi1 Bi 0.11 0.22 0.33 1.0", "O1 O 0.61 0.42 0.13 1.0"]),
}


@pytest.fixture(scope="module")
def structures():
    return {name: _build(*args) for name, args in SYSTEMS.items()}


@pytest.fixture(scope="module")
def cubic(structures):
    return structures["cubic"]


def _polygon_area(polygon) -> float:
    if len(polygon) < 3:
        return 0.0
    centroid = polygon.mean(axis=0)
    total = np.zeros(3)
    for i in range(len(polygon)):
        total = total + np.cross(polygon[i] - centroid,
                                 polygon[(i + 1) % len(polygon)] - centroid)
    return float(np.linalg.norm(total)) / 2.0


# ===========================================================================
# the definition of a plane
# ===========================================================================

def test_spacing_agrees_with_the_diffraction_code(structures):
    """One definition of d in the program, not two."""
    for structure in structures.values():
        for hkl in HKL:
            _, d = P.normal_and_spacing(structure, *hkl)
            assert d == pytest.approx(d_spacing(structure, *hkl), rel=1e-12)


def test_the_normal_is_perpendicular_to_every_in_plane_lattice_vector(structures):
    """A lattice vector [uvw] lies in (hkl) exactly when hu + kv + lw = 0.

    The plane's normal must then be perpendicular to it, in every cell. This is
    the test that distinguishes the reciprocal-lattice normal from the
    real-space combination, and it is checked over every [uvw] with indices to
    three -- a few hundred vectors per plane.
    """
    for name, structure in structures.items():
        orth = structure.cell.orth
        for hkl in HKL:
            normal, _ = P.normal_and_spacing(structure, *hkl)
            for uvw in itertools.product(range(-3, 4), repeat=3):
                if uvw == (0, 0, 0):
                    continue
                if sum(a * b for a, b in zip(hkl, uvw)) != 0:
                    continue
                vector = orth @ np.array(uvw, float)
                cosine = abs(float(np.dot(normal, vector))
                             / np.linalg.norm(vector))
                assert cosine < 1e-12, f"{name} {hkl} not normal to {uvw}"


def test_the_real_space_combination_really_is_wrong(structures):
    """The mistake this module exists to avoid, measured.

    ``h a + k b + l c`` agrees with the true normal only in a cubic cell. Pinned
    so that anyone tempted to simplify sees the size of the error first.
    """
    def disagreement(structure, hkl):
        normal, _ = P.normal_and_spacing(structure, *hkl)
        naive = structure.cell.orth @ np.array(hkl, float)
        naive = naive / np.linalg.norm(naive)
        if float(np.dot(normal, naive)) < 0:
            naive = -naive
        # atan2, not acos: acos of a dot product near 1 keeps only half its
        # digits, and the cubic case below is exactly that
        return math.degrees(math.atan2(
            float(np.linalg.norm(np.cross(normal, naive))),
            float(np.dot(normal, naive))))

    # cubic: the two agree, so a cubic-only test would never notice
    for hkl in HKL:
        assert disagreement(structures["cubic"], hkl) < 1e-9

    assert disagreement(structures["hexagonal"], (1, 1, 1)) > 50.0
    assert disagreement(structures["monoclinic"], (1, -1, 2)) > 20.0
    assert disagreement(structures["triclinic"], (3, 1, -2)) > 15.0


def test_zero_indices_are_rejected(cubic):
    with pytest.raises(ValueError, match="not a plane"):
        P.normal_and_spacing(cubic, 0, 0, 0)
    assert not P.LatticePlane(0, 0, 0).is_valid
    assert P.plane_polygons(cubic, P.LatticePlane(0, 0, 0)) == []


def test_in_plane_axes_are_orthonormal_and_in_the_plane(structures):
    for structure in structures.values():
        for hkl in HKL:
            u, v, normal = P.in_plane_axes(structure, *hkl)
            for vector in (u, v, normal):
                assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-9)
            assert abs(float(np.dot(u, v))) < 1e-9
            assert abs(float(np.dot(u, normal))) < 1e-9
            assert abs(float(np.dot(v, normal))) < 1e-9
            # right-handed
            assert np.allclose(np.cross(u, v), normal, atol=1e-9)


def test_interplanar_angles_match_the_textbook_cubic_values(cubic):
    cases = [((1, 0, 0), (0, 1, 0), 90.0),
             ((1, 0, 0), (1, 1, 0), 45.0),
             ((1, 1, 1), (1, 1, -1), 70.528779),
             ((1, 0, 0), (1, 1, 1), 54.735610),
             ((1, 1, 0), (1, 1, 1), 35.264390)]
    for first, second, expected in cases:
        assert P.interplanar_angle(cubic, first, second) == pytest.approx(
            expected, abs=1e-5)


def test_monoclinic_interplanar_angle_has_a_closed_form(structures):
    """In a monoclinic cell the angle between (100) and (001) is 180 - beta."""
    structure = structures["monoclinic"]
    assert P.interplanar_angle(structure, (1, 0, 0), (0, 0, 1)) == \
        pytest.approx(180.0 - 113.0, abs=1e-6)


def test_interplanar_angle_of_a_plane_with_itself_is_zero(structures):
    for structure in structures.values():
        for hkl in HKL:
            assert P.interplanar_angle(structure, hkl, hkl) == pytest.approx(
                0.0, abs=1e-6)


# ===========================================================================
# cutting the box
# ===========================================================================

def test_cross_section_areas_match_the_closed_form(cubic):
    """Four sections of a cube whose areas can be written down.

    (001) and (100) through the middle are squares of side a. (110) through the
    body diagonal is a rectangle of a by a*sqrt(2). (111) placed a third of the
    way across cuts off an equilateral triangle of area sqrt(3)/2 a^2.
    """
    a = 5.6402
    corners = P.box_corners(cubic, (1, 1, 1), margin=0.0)
    cases = [((0, 0, 1), 0.5, a * a, 4),
             ((1, 0, 0), 0.5, a * a, 4),
             ((1, 1, 0), 0.5, a * a * math.sqrt(2.0), 4),
             ((1, 1, 1), 1.0 / 3.0, math.sqrt(3.0) / 2.0 * a * a, 3)]
    for hkl, fraction, expected, vertices in cases:
        normal, _ = P.normal_and_spacing(cubic, *hkl)
        span = corners @ normal
        point = normal * (span.min() + fraction * (span.max() - span.min()))
        polygon = P.polygon_in_box(normal, point, corners)
        assert len(polygon) == vertices
        assert _polygon_area(polygon) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("system", sorted(SYSTEMS))
def test_every_cross_section_is_planar_convex_and_ordered(structures, system):
    """The three properties the triangle fan and the blend both depend on.

    Planar: every vertex the same distance along the normal. Convex and in
    order: every consecutive edge pair turns the same way. Checked at nine
    positions across the box for ten plane families.
    """
    structure = structures[system]
    corners = P.box_corners(structure, (1, 1, 1), margin=0.35)
    checked = 0
    for hkl in HKL:
        normal, _ = P.normal_and_spacing(structure, *hkl)
        span = corners @ normal
        for fraction in np.linspace(0.02, 0.98, 9):
            point = normal * (span.min()
                              + fraction * (span.max() - span.min()))
            polygon = P.polygon_in_box(normal, point, corners)
            if len(polygon) < 3:
                continue
            checked += 1

            along = polygon @ normal
            assert float(along.max() - along.min()) < 1e-8, "not planar"

            turns = []
            for i in range(len(polygon)):
                first = polygon[(i + 1) % len(polygon)] - polygon[i]
                second = (polygon[(i + 2) % len(polygon)]
                          - polygon[(i + 1) % len(polygon)])
                turns.append(float(np.dot(np.cross(first, second), normal)))
            turns = np.array(turns)
            significant = turns[np.abs(turns) > 1e-9]
            if len(significant):
                assert (np.all(significant > 0) or np.all(significant < 0)), \
                    f"{system} {hkl} at {fraction}: not convex or mis-ordered"
    assert checked > 50


def test_a_plane_that_misses_the_box_gives_no_polygon(cubic):
    normal, d = P.normal_and_spacing(cubic, 0, 0, 1)
    corners = P.box_corners(cubic, (1, 1, 1))
    far = P.polygon_in_box(normal, normal * 500.0, corners)
    assert len(far) == 0


def test_triangulation_preserves_the_area_and_the_normal(structures):
    for structure in structures.values():
        corners = P.box_corners(structure, (1, 1, 1), margin=0.0)
        for hkl in HKL[:6]:
            normal, _ = P.normal_and_spacing(structure, *hkl)
            span = corners @ normal
            point = normal * (span.min() + 0.5 * (span.max() - span.min()))
            polygon = P.polygon_in_box(normal, point, corners)
            if len(polygon) < 3:
                continue
            vertices, normals = P.triangulate(polygon)
            assert len(vertices) == 3 * (len(polygon) - 2)
            total = 0.0
            for i in range(0, len(vertices), 3):
                total += float(np.linalg.norm(np.cross(
                    vertices[i + 1] - vertices[i],
                    vertices[i + 2] - vertices[i]))) / 2.0
            assert total == pytest.approx(_polygon_area(polygon), rel=1e-6)
            assert np.allclose(np.abs(normals @ normal), 1.0, atol=1e-5)


def test_triangulating_nothing_gives_nothing():
    vertices, normals = P.triangulate(np.zeros((2, 3)))
    assert len(vertices) == 0 and len(normals) == 0


def test_the_margined_box_is_still_a_parallelepiped(structures):
    """The margin must not bend the box.

    Pushing the corners outwards radially would grow the four body diagonals by
    different fractions, the faces would stop being planar, and a cross-section
    would no longer be guaranteed convex -- which everything downstream assumes.
    Opposite edges of a parallelepiped are equal and parallel; that is asserted
    here directly.
    """
    for structure in structures.values():
        corners = P.box_corners(structure, (2, 1, 3), margin=0.8)
        # index = x | y<<1 | z<<2, so edges along x are (i, i^1) and so on
        for bit in (1, 2, 4):
            vectors = [corners[i ^ bit] - corners[i]
                       for i in range(8) if not i & bit]
            for vector in vectors[1:]:
                assert np.allclose(vector, vectors[0], atol=1e-9)


def test_the_margin_grows_the_box(structures):
    for structure in structures.values():
        tight = P.box_corners(structure, (1, 1, 1), margin=0.0)
        loose = P.box_corners(structure, (1, 1, 1), margin=0.5)
        centre = tight.mean(axis=0)
        assert np.allclose(loose.mean(axis=0), centre, atol=1e-9)
        for a, b in zip(tight, loose):
            assert np.linalg.norm(b - centre) > np.linalg.norm(a - centre)


def test_repeat_places_planes_one_d_apart(cubic):
    plane = P.LatticePlane(0, 0, 1, repeat=4)
    polygons = P.plane_polygons(cubic, plane, (1, 1, 1), margin=0.0)
    normal, d = P.normal_and_spacing(cubic, 0, 0, 1)
    positions = sorted(float(np.mean(polygon @ normal)) for polygon in polygons)
    assert len(positions) >= 2
    gaps = np.diff(positions)
    assert np.allclose(gaps, d, atol=1e-6)


def test_the_cell_range_widens_the_planes(cubic):
    one = P.plane_polygons(cubic, P.LatticePlane(0, 0, 1, offset=0.5),
                           (1, 1, 1), margin=0.0)
    many = P.plane_polygons(cubic, P.LatticePlane(0, 0, 1, offset=0.5),
                            (3, 3, 1), margin=0.0)
    assert _polygon_area(many[0]) == pytest.approx(
        9.0 * _polygon_area(one[0]), rel=1e-6)


# ===========================================================================
# what lies on a plane
# ===========================================================================

def test_rocksalt_002_planes_hold_every_atom(cubic):
    """In rocksalt the (002) planes pass through every atom.

    Na sits at z = 0 and 1/2; Cl at z = 0 and 1/2 as well, in the F lattice. So
    with d(002) = a/2, every atom of the cell lies on a plane of the family.
    """
    plane = P.LatticePlane(0, 0, 2)
    on = P.atoms_on_plane(cubic, plane, tolerance=1e-6)
    assert len(on) == len(cubic.atoms)


def test_rocksalt_001_planes_hold_only_the_z_zero_atoms(cubic):
    plane = P.LatticePlane(0, 0, 1)
    on = P.atoms_on_plane(cubic, plane, tolerance=1e-6)
    expected = [i for i, atom in enumerate(cubic.atoms)
                if abs(atom.frac[2] % 1.0) < 1e-6]
    assert sorted(on) == sorted(expected)
    assert len(on) == 4


def test_an_offset_moves_which_atoms_are_on_the_plane(cubic):
    at_origin = P.atoms_on_plane(cubic, P.LatticePlane(0, 0, 1), 1e-6)
    halfway = P.atoms_on_plane(cubic, P.LatticePlane(0, 0, 1, offset=0.5), 1e-6)
    assert at_origin and halfway
    assert set(at_origin) != set(halfway)
    assert not set(at_origin) & set(halfway)


def test_plane_occupancy_counts_by_element(cubic):
    composition = P.plane_occupancy(cubic, P.LatticePlane(0, 0, 2), 1e-6)
    assert composition == {"Na": pytest.approx(4.0), "Cl": pytest.approx(4.0)}


def test_a_wider_tolerance_can_only_add_atoms(structures):
    for structure in structures.values():
        plane = P.LatticePlane(1, 0, 1)
        previous: set[int] = set()
        for tolerance in (0.01, 0.1, 0.3, 1.0, 3.0):
            found = set(P.atoms_on_plane(structure, plane, tolerance))
            assert previous <= found
            previous = found


def test_signed_distances_change_sign_across_the_plane(cubic):
    normal, d = P.normal_and_spacing(cubic, 0, 0, 1)
    points = np.array([normal * -1.0, normal * 1.0, np.zeros(3)])
    distances = P.signed_distances(cubic, points, 0, 0, 1)
    assert distances[0] < 0 < distances[1]
    assert distances[2] == pytest.approx(0.0, abs=1e-9)


# ===========================================================================
# slabs
# ===========================================================================

def test_a_disabled_slab_keeps_everything(cubic):
    slab = P.Slab(0, 0, 1, thickness=0.01, enabled=False)
    assert P.slab_mask(cubic, cubic.cart_array(), slab).all()


def test_a_thick_slab_keeps_everything_and_a_thin_one_keeps_less(cubic):
    everything = P.Slab(0, 0, 1, thickness=1000.0, enabled=True)
    assert P.slab_mask(cubic, cubic.cart_array(), everything).all()

    thin = P.Slab(0, 0, 1, thickness=0.05, enabled=True)
    kept = P.slab_mask(cubic, cubic.cart_array(), thin)
    assert 0 < kept.sum() < len(cubic.atoms)


def test_the_slab_thickness_is_in_angstrom_along_the_normal(cubic):
    """An atom exactly at half the thickness is in; one just beyond is out."""
    normal, _ = P.normal_and_spacing(cubic, 0, 0, 1)
    slab = P.Slab(0, 0, 1, centre=0.0, thickness=2.0, enabled=True)
    points = np.array([normal * 0.99, normal * 1.0, normal * 1.01,
                       normal * -0.99, normal * -1.01])
    mask = P.slab_mask(cubic, points, slab)
    assert list(mask) == [True, True, False, True, False]


def test_moving_the_slab_centre_moves_the_selection(cubic):
    low = P.Slab(0, 0, 1, centre=0.0, thickness=1.0, enabled=True)
    high = P.Slab(0, 0, 1, centre=0.5, thickness=1.0, enabled=True)
    assert set(P.atoms_in_slab(cubic, low)) != set(P.atoms_in_slab(cubic, high))


def test_an_invalid_slab_keeps_everything(cubic):
    for slab in (P.Slab(0, 0, 0, enabled=True),
                 P.Slab(0, 0, 1, thickness=0.0, enabled=True)):
        assert not slab.is_valid
        assert P.slab_mask(cubic, cubic.cart_array(), slab).all()


# ===========================================================================
# the slab applied to a scene: the index remapping
# ===========================================================================

@pytest.fixture(scope="module")
def bi_structure():
    """A real structure with enough atoms for the remapping to matter."""
    path = Path(r"C:\Users\samso\Desktop\WSU_work\XRD\cif\Bi\cifs"
                r"\1526458_Bi2O3.cif")
    if not path.is_file():
        pytest.skip("the Bi CIF collection is not present")
    return cif.read(path)


def test_the_slab_filter_keeps_the_scene_self_consistent(bi_structure):
    """Atoms are renumbered, so everything indexed by atom must move with them.

    The check that matters: for every bond that survives, the two atoms it now
    points at must be the same two atoms it pointed at before -- compared by
    position, not by index, because the indices are exactly what changed.
    """
    from facet.gl.scene import apply_slab, build_scene

    scene = build_scene(bi_structure, cell_range=(2, 2, 1))
    before_positions = scene.atom_position.copy()
    before_pairs = scene.bond_atoms.copy()
    before_endpoints = {
        tuple(np.round(scene.bond_a[i], 4)) + tuple(np.round(scene.bond_b[i], 4))
        for i in range(scene.n_bonds)}

    slab = P.Slab(0, 0, 1, centre=0.25, thickness=3.0, enabled=True)
    apply_slab(bi_structure, scene, slab)

    assert 0 < scene.n_atoms < len(before_positions), "the slab cut nothing"

    # every surviving atom is one that was inside the slab
    mask = P.slab_mask(bi_structure, scene.atom_position, slab)
    assert mask.all(), "an atom outside the slab survived"

    # no atom that was inside was dropped
    was_inside = P.slab_mask(bi_structure, before_positions, slab).sum()
    assert scene.n_atoms == was_inside

    # every surviving bond was there before, geometrically unchanged
    for i in range(scene.n_bonds):
        key = (tuple(np.round(scene.bond_a[i], 4))
               + tuple(np.round(scene.bond_b[i], 4)))
        assert key in before_endpoints, "the filter invented a bond"
    for a, b in scene.bond_atoms:
        assert 0 <= a < scene.n_atoms and 0 <= b < scene.n_atoms

    # arrays indexed by atom all have the same length
    assert len(scene.atom_radius) == scene.n_atoms
    assert len(scene.atom_color) == scene.n_atoms
    assert len(scene.atom_index) == scene.n_atoms
    assert len(scene.atom_site) == scene.n_atoms
    assert len(scene.atom_label) == scene.n_atoms
    assert len(scene.atom_element) == scene.n_atoms

    # arrays indexed by bond all have the same length
    n = scene.n_bonds
    for array in (scene.bond_a, scene.bond_b, scene.bond_color_a,
                  scene.bond_color_b, scene.bond_valence,
                  scene.bond_distance, scene.bond_atoms,
                  scene._bond_base_a, scene._bond_base_b):
        assert len(array) == n

    assert scene.n_cell_atoms <= scene.n_atoms


@pytest.mark.parametrize("cell_range", [(1, 1, 1), (2, 1, 1), (2, 2, 1),
                                        (2, 2, 2), (3, 1, 2)])
def test_every_bond_array_has_the_same_length_at_every_cell_range(bi_structure,
                                                                  cell_range):
    """The invariant a shape mismatch in the slab filter exposed.

    _replicate tiles most bond arrays but deliberately leaves the colours to
    restyle, so between those two steps a replicated scene is internally
    inconsistent -- the colour arrays are one cell long while everything else is
    n_copies longer. Anything that filters bonds has to run after restyle, and
    this asserts the state a caller is entitled to assume.
    """
    from facet.gl.scene import build_scene

    scene = build_scene(bi_structure, cell_range=cell_range)
    n = scene.n_bonds
    for name in ("bond_a", "bond_b", "bond_radius", "bond_color_a",
                 "bond_color_b", "bond_valence", "bond_distance",
                 "bond_atoms", "_bond_base_a", "_bond_base_b"):
        assert len(getattr(scene, name)) == n, f"{name} is out of step"
    for name in ("atom_position", "atom_radius", "atom_color", "atom_index",
                 "atom_site", "atom_label", "atom_element"):
        assert len(getattr(scene, name)) == scene.n_atoms, \
            f"{name} is out of step"


@pytest.mark.parametrize("cell_range", [(1, 1, 1), (2, 2, 1), (2, 2, 2)])
def test_the_slab_leaves_every_array_in_step(bi_structure, cell_range):
    from facet.gl.scene import build_scene

    scene = build_scene(
        bi_structure, cell_range=cell_range,
        slab=P.Slab(0, 0, 1, centre=0.2, thickness=3.0, enabled=True))
    n = scene.n_bonds
    for name in ("bond_a", "bond_b", "bond_radius", "bond_color_a",
                 "bond_color_b", "bond_valence", "bond_distance",
                 "bond_atoms", "_bond_base_a", "_bond_base_b"):
        assert len(getattr(scene, name)) == n, f"{name} is out of step"
    for name in ("atom_position", "atom_radius", "atom_color", "atom_index",
                 "atom_site", "atom_label", "atom_element"):
        assert len(getattr(scene, name)) == scene.n_atoms, \
            f"{name} is out of step"


def test_a_bond_reaching_out_of_the_slab_is_removed(bi_structure):
    """A bond with one end outside must go, or it trails into nothing."""
    from facet.gl.scene import apply_slab, build_scene

    scene = build_scene(bi_structure)
    slab = P.Slab(0, 0, 1, centre=0.0, thickness=1.6, enabled=True)
    apply_slab(bi_structure, scene, slab)
    for a, b in scene.bond_atoms:
        for end in (scene.atom_position[a], scene.atom_position[b]):
            assert P.slab_mask(bi_structure, end.reshape(1, 3), slab)[0]


def test_every_drawn_bond_endpoint_sits_on_a_drawn_atom(bi_structure):
    """No tube may end in empty space, before or after a slab.

    This is the property the earlier "bonds trailing to undrawn atoms" fault
    violated, and a slab can break it again. The two endpoints are not
    symmetrical: bond_a is exactly the first named atom's position, but bond_b is
    the *contact* position, which for a bond crossing the cell edge is a periodic
    image -- while bond_atoms names that image's home-cell representative, which
    can be ten angstrom away. A filter that judged bonds by bond_atoms would keep
    bonds whose drawn far end had been cut off.
    """
    from scipy.spatial import cKDTree

    from facet.gl.scene import apply_slab, build_scene

    for slab in (None,
                 P.Slab(0, 0, 1, centre=0.1, thickness=4.0, enabled=True),
                 P.Slab(0, 0, 1, centre=0.0, thickness=1.6, enabled=True),
                 P.Slab(1, 0, 1, centre=0.3, thickness=3.0, enabled=True)):
        scene = build_scene(bi_structure, cell_range=(2, 1, 1))
        if slab is not None:
            apply_slab(bi_structure, scene, slab)
        if not scene.n_bonds:
            continue
        tree = cKDTree(scene.atom_position)
        for array in (scene.bond_a, scene.bond_b):
            distance, _ = tree.query(np.asarray(array, float))
            assert float(distance.max()) < 1e-4, \
                "a bond ends where no atom is drawn"


def test_the_slab_names_bond_atoms_that_are_still_drawn(bi_structure):
    """After filtering, bond_atoms must index surviving atoms.

    And each named atom must be the one actually at that end of the tube, so a
    click on a bond names what it looks like it names.
    """
    from facet.gl.scene import apply_slab, build_scene

    scene = build_scene(bi_structure, cell_range=(2, 1, 1))
    apply_slab(bi_structure, scene,
               P.Slab(0, 0, 1, centre=0.1, thickness=4.0, enabled=True))
    for i, (a, b) in enumerate(scene.bond_atoms):
        assert 0 <= a < scene.n_atoms
        assert 0 <= b < scene.n_atoms
        assert np.allclose(scene.atom_position[a], scene.bond_a[i], atol=1e-4)
        assert np.allclose(scene.atom_position[b], scene.bond_b[i], atol=1e-4)


def test_bond_distance_still_matches_the_drawn_length(bi_structure):
    from facet.gl.scene import apply_slab, build_scene

    scene = build_scene(bi_structure, cell_range=(2, 1, 1))
    apply_slab(bi_structure, scene,
               P.Slab(0, 0, 1, centre=0.1, thickness=4.0, enabled=True))
    if not scene.n_bonds:
        pytest.skip("the slab removed every bond")
    drawn = np.linalg.norm(scene.bond_b - scene.bond_a, axis=1)
    assert np.allclose(drawn, scene.bond_distance, atol=1e-4)


def test_the_selected_atom_follows_the_renumbering(bi_structure):
    from facet.gl.scene import apply_slab, build_scene

    scene = build_scene(bi_structure)
    slab = P.Slab(0, 0, 1, centre=0.25, thickness=3.0, enabled=True)
    inside = np.nonzero(P.slab_mask(bi_structure, scene.atom_position,
                                    slab))[0]
    assert len(inside) >= 2

    chosen = int(inside[-1])
    position = scene.atom_position[chosen].copy()
    scene.selected_atom = chosen
    apply_slab(bi_structure, scene, slab)
    assert scene.selected_atom is not None
    assert np.allclose(scene.atom_position[scene.selected_atom], position)


def test_a_selection_cut_away_is_cleared(bi_structure):
    from facet.gl.scene import apply_slab, build_scene

    scene = build_scene(bi_structure)
    slab = P.Slab(0, 0, 1, centre=0.0, thickness=1.0, enabled=True)
    outside = np.nonzero(~P.slab_mask(bi_structure, scene.atom_position,
                                      slab))[0]
    if not len(outside):
        pytest.skip("this structure has no atom outside that slab")
    scene.selected_atom = int(outside[0])
    apply_slab(bi_structure, scene, slab)
    assert scene.selected_atom is None


def test_a_slab_that_keeps_everything_changes_nothing(bi_structure):
    from facet.gl.scene import apply_slab, build_scene

    scene = build_scene(bi_structure)
    before = (scene.n_atoms, scene.n_bonds, scene.n_cell_atoms,
              scene.atom_position.copy(), scene.bond_atoms.copy())
    apply_slab(bi_structure, scene,
               P.Slab(0, 0, 1, thickness=1000.0, enabled=True))
    assert scene.n_atoms == before[0]
    assert scene.n_bonds == before[1]
    assert scene.n_cell_atoms == before[2]
    assert np.array_equal(scene.atom_position, before[3])
    assert np.array_equal(scene.bond_atoms, before[4])


def test_polyhedron_triangles_outside_the_slab_are_removed(bi_structure):
    from facet.gl.scene import apply_slab, build_scene

    scene = build_scene(bi_structure, polyhedron_sites=list(
        range(len(bi_structure.sites))))
    if not scene.n_poly_triangles:
        pytest.skip("no polyhedra were built")
    before = scene.n_poly_triangles
    slab = P.Slab(0, 0, 1, centre=0.0, thickness=1.5, enabled=True)
    apply_slab(bi_structure, scene, slab)
    assert scene.n_poly_triangles <= before
    assert len(scene.poly_normals) == len(scene.poly_vertices)
    assert len(scene.poly_vertices) % 3 == 0


# ===========================================================================
# planes in a scene
# ===========================================================================

def test_build_scene_draws_the_requested_planes(cubic):
    from facet.gl.scene import build_scene

    scene = build_scene(cubic, lattice_planes=[
        P.LatticePlane(0, 0, 1, offset=0.5, color=(1.0, 0.0, 0.0), alpha=0.4),
        P.LatticePlane(1, 1, 0, offset=0.5, color=(0.0, 1.0, 0.0), alpha=0.2)])
    assert scene.n_planes == 2
    colours = [mesh[2] for mesh in scene.plane_meshes]
    alphas = [mesh[3] for mesh in scene.plane_meshes]
    assert (1.0, 0.0, 0.0) in colours and (0.0, 1.0, 0.0) in colours
    assert 0.4 in alphas and 0.2 in alphas
    for vertices, normals, _, _ in scene.plane_meshes:
        assert len(vertices) >= 3
        assert len(vertices) % 3 == 0
        assert len(normals) == len(vertices)
    assert len(scene.plane_edges) > 0


def test_a_hidden_plane_is_not_built(cubic):
    from facet.gl.scene import build_scene

    scene = build_scene(cubic, lattice_planes=[
        P.LatticePlane(0, 0, 1, offset=0.5, visible=False)])
    assert scene.n_planes == 0


def test_planes_are_included_in_the_framing(cubic):
    """A plane drawn across the box must not fall outside the camera's sphere."""
    from facet.gl.scene import build_scene

    scene = build_scene(cubic, lattice_planes=[
        P.LatticePlane(0, 0, 1, offset=0.5, repeat=3)])
    assert scene.n_planes >= 1
    for vertices, _, _, _ in scene.plane_meshes:
        distances = np.linalg.norm(vertices - scene.center, axis=1)
        assert float(distances.max()) <= scene.radius + 1e-4


def test_planes_are_not_replicated_across_the_cell_range(cubic):
    """A plane is already cut to the whole drawn box.

    Replicating it would stack coincident sheets and darken the blend, which
    reads as a different colour for no stated reason.
    """
    from facet.gl.scene import build_scene

    one = build_scene(cubic, lattice_planes=[
        P.LatticePlane(0, 0, 1, offset=0.5)], cell_range=(1, 1, 1))
    many = build_scene(cubic, lattice_planes=[
        P.LatticePlane(0, 0, 1, offset=0.5)], cell_range=(3, 3, 2))
    assert one.n_planes == many.n_planes == 1
    # and the one plane is larger, because the box is
    def area(scene):
        vertices = scene.plane_meshes[0][0].reshape(-1, 3, 3)
        return sum(float(np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0]))) / 2
                   for t in vertices)
    assert area(many) > 5.0 * area(one)


def test_a_slab_does_not_cut_the_planes_that_show_it(cubic):
    """The plane marking a slab boundary must survive the slab."""
    from facet.gl.scene import build_scene

    scene = build_scene(
        cubic,
        lattice_planes=[P.LatticePlane(0, 0, 1, offset=0.5)],
        slab=P.Slab(0, 0, 1, centre=0.5, thickness=1.0, enabled=True))
    assert scene.n_planes == 1
    assert len(scene.plane_meshes[0][0]) >= 3
