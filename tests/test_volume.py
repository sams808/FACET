"""Grids, isosurfaces, sections and bond-valence maps.

The isosurface is checked against a sphere, whose area, volume and normals are
known exactly. A triangulator that is subtly wrong still produces a plausible
picture, so a picture is not a check.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from conftest import sample_cif

from facet.core import cif, coordination, volume
from facet.core.readers import cell_from_parameters

CIFS = Path(r"C:\Users\samso\Desktop\WSU_work\XRD\cif\Bi\cifs")
SAMPLE = sample_cif("1526458", "1526458_Bi2O3.cif")


def sphere_grid(a: float = 10.0, radius: float = 3.0, n: int = 40):
    """f = R - |r - centre|, so f = 0 is a sphere of radius R."""
    cell = cell_from_parameters(a, a, a, 90, 90, 90)
    ticks = np.arange(n) / n
    fa, fb, fc = np.meshgrid(ticks, ticks, ticks, indexing="ij")
    cart = np.stack([fa, fb, fc], axis=-1) * a
    centre = np.array([a / 2, a / 2, a / 2])
    values = radius - np.linalg.norm(cart - centre, axis=-1)
    return volume.Grid(values, cell, name="sphere"), centre, radius


# --- the grid ----------------------------------------------------------------

def test_grid_reports_its_shape_and_spacing():
    grid, _, _ = sphere_grid(a=10.0, n=20)
    assert grid.shape == (20, 20, 20)
    assert grid.n_points == 8000
    assert all(s == pytest.approx(0.5) for s in grid.spacing)


def test_interpolation_reproduces_the_samples_exactly():
    """At a grid node, trilinear interpolation must return the stored value."""
    grid, _, _ = sphere_grid(n=12)
    for index in [(0, 0, 0), (3, 7, 5), (11, 11, 11)]:
        frac = np.array(index) / 12.0
        assert grid.at_fractional([frac])[0] == pytest.approx(
            grid.values[index], rel=1e-12)


def test_interpolation_is_periodic():
    grid, _, _ = sphere_grid(n=16)
    inside = grid.at_fractional([[0.3, 0.4, 0.5]])[0]
    wrapped = grid.at_fractional([[1.3, -0.6, 2.5]])[0]
    assert inside == pytest.approx(wrapped, rel=1e-12)


def test_interpolation_is_smooth_between_nodes():
    grid, centre, radius = sphere_grid(n=40)
    # halfway between two nodes the field should be close to the analytic value
    frac = np.array([0.5, 0.5, 0.3125])
    cart = grid.cell.to_cartesian(frac)
    analytic = radius - np.linalg.norm(cart - centre)
    assert grid.at_fractional([frac])[0] == pytest.approx(analytic, abs=0.05)


def test_a_difference_needs_the_same_mesh():
    a, _, _ = sphere_grid(n=16)
    b, _, _ = sphere_grid(n=20)
    with pytest.raises(ValueError):
        a.difference(b)


def test_a_difference_subtracts():
    a, _, _ = sphere_grid(n=12)
    b, _, _ = sphere_grid(n=12)
    assert np.allclose(a.difference(b).values, 0.0)


# --- isosurface --------------------------------------------------------------

class TestIsosurfaceAgainstASphere:

    @pytest.fixture(scope="class")
    def surface(self):
        grid, centre, radius = sphere_grid(a=10.0, radius=3.0, n=40)
        vertices, faces, normals = volume.isosurface(grid, 0.0)
        return vertices, faces, normals, centre, radius

    def test_it_produces_a_surface(self, surface):
        vertices, faces, _, _, _ = surface
        assert len(vertices) > 1000
        assert len(faces) > 1000

    def test_every_vertex_lies_on_the_sphere(self, surface):
        vertices, _, _, centre, radius = surface
        r = np.linalg.norm(vertices - centre, axis=1)
        assert r.mean() == pytest.approx(radius, abs=0.01)
        assert r.std() < 0.01

    def test_the_area_matches_four_pi_r_squared(self, surface):
        vertices, faces, _, _, radius = surface
        area = 0.0
        for i, j, k in faces:
            area += 0.5 * np.linalg.norm(
                np.cross(vertices[j] - vertices[i], vertices[k] - vertices[i]))
        assert area == pytest.approx(4 * math.pi * radius ** 2, rel=0.02)

    def test_the_enclosed_volume_matches_the_sphere(self, surface):
        vertices, faces, _, centre, radius = surface
        total = 0.0
        for i, j, k in faces:
            a, b, c = (vertices[x] - centre for x in (i, j, k))
            total += abs(float(np.dot(a, np.cross(b, c)))) / 6.0
        assert total == pytest.approx(4 / 3 * math.pi * radius ** 3, rel=0.02)

    def test_the_normals_point_radially_outwards(self, surface):
        vertices, _, normals, centre, _ = surface
        radial = vertices - centre
        radial /= np.linalg.norm(radial, axis=1)[:, None]
        assert (normals * radial).sum(axis=1).mean() == pytest.approx(
            1.0, abs=0.01)

    def test_the_normals_are_unit_length(self, surface):
        _, _, normals, _, _ = surface
        assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-9)

    def test_every_face_indexes_real_vertices(self, surface):
        vertices, faces, _, _, _ = surface
        assert faces.min() >= 0
        assert faces.max() < len(vertices)

    def test_no_face_is_degenerate(self, surface):
        vertices, faces, _, _, _ = surface
        for i, j, k in faces[:400]:
            assert len({i, j, k}) == 3

    def test_shared_edges_share_a_vertex(self, surface):
        """Marching tetrahedra caches by grid edge, so adjacent tetrahedra
        reuse a vertex and the mesh is watertight rather than loose
        triangles."""
        vertices, faces, _, _, _ = surface
        assert len(vertices) < 3 * len(faces)


def test_a_level_outside_the_field_gives_nothing():
    grid, _, _ = sphere_grid()
    vertices, faces, _ = volume.isosurface(grid, 1000.0)
    assert len(vertices) == 0 and len(faces) == 0


def test_subsampling_gives_a_coarser_surface_in_the_same_place():
    grid, centre, radius = sphere_grid(n=40)
    fine, _, _ = volume.isosurface(grid, 0.0, step=1)
    coarse, _, _ = volume.isosurface(grid, 0.0, step=2)
    assert len(coarse) < len(fine)
    r = np.linalg.norm(coarse - centre, axis=1)
    assert r.mean() == pytest.approx(radius, abs=0.05)


def test_tiling_repeats_the_surface():
    grid, _, _ = sphere_grid(n=20)
    one, faces_one, _ = volume.isosurface(grid, 0.0)
    many, faces_many, _ = volume.isosurface(grid, 0.0, cell_range=(2, 1, 1))
    assert len(many) == 2 * len(one)
    assert len(faces_many) == 2 * len(faces_one)


# --- sections and planes -----------------------------------------------------

def test_a_section_samples_the_field_on_a_plane():
    grid, centre, radius = sphere_grid(n=40)
    values, extent = volume.section(grid, centre, [1, 0, 0], [0, 1, 0],
                                    size=8.0, samples=64)
    assert values.shape == (64, 64)
    assert extent == (-4.0, 4.0, -4.0, 4.0)
    # the centre of the plane passes through the centre of the sphere
    assert values[32, 32] == pytest.approx(radius, abs=0.2)


def test_a_section_refuses_parallel_axes():
    grid, centre, _ = sphere_grid(n=12)
    with pytest.raises(ValueError):
        volume.section(grid, centre, [1, 0, 0], [2, 0, 0])


def test_miller_plane_normal_is_reciprocal_not_real_space():
    """The (hkl) normal is h a* + k b* + l c*. Using h a + k b + l c is the
    same thing only in a cubic cell, and is a standard way to get an oblique
    structure's planes wrong."""
    cell = cell_from_parameters(5.0, 5.0, 5.0, 90, 90, 120)
    u, v, normal = volume.miller_plane_axes(cell, 1, 0, 0)

    a = cell.orth[:, 0]
    assert not np.allclose(normal, a / np.linalg.norm(a), atol=0.01)
    # the normal must be perpendicular to b and c, which lie in the (100) plane
    for axis in (1, 2):
        assert float(np.dot(normal, cell.orth[:, axis])) == pytest.approx(
            0.0, abs=1e-9)


def test_miller_plane_axes_are_orthonormal():
    cell = cell_from_parameters(4.1, 7.3, 9.9, 71.0, 103.0, 118.0)
    u, v, normal = volume.miller_plane_axes(cell, 1, 1, 2)
    for vector in (u, v, normal):
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-9)
    assert float(np.dot(u, v)) == pytest.approx(0.0, abs=1e-9)
    assert float(np.dot(u, normal)) == pytest.approx(0.0, abs=1e-9)


def test_the_zero_plane_is_refused():
    cell = cell_from_parameters(5, 5, 5, 90, 90, 90)
    with pytest.raises(ValueError):
        volume.miller_plane_axes(cell, 0, 0, 0)


def test_cubic_plane_normals_are_the_axes():
    cell = cell_from_parameters(5, 5, 5, 90, 90, 90)
    _, _, normal = volume.miller_plane_axes(cell, 1, 0, 0)
    assert np.allclose(np.abs(normal), [1, 0, 0], atol=1e-9)


# --- bond-valence maps -------------------------------------------------------

@pytest.mark.skipif(not SAMPLE.exists(), reason="sample structure not present")
class TestBondValenceMap:

    @pytest.fixture(scope="class")
    def mapped(self):
        structure = cif.read(SAMPLE)
        results = coordination.analyse_structure(structure)
        grid = volume.bond_valence_grid(structure, "Bi", 3, resolution=0.25)
        return structure, results, grid

    def test_the_map_reproduces_the_site_analysis(self, mapped):
        """Sampled at a real cation site, the map must return roughly that
        site's own bond-valence sum -- computed by a completely different
        route."""
        structure, results, grid = mapped
        for r in results:
            site = structure.sites[r.site_index]
            sampled = float(grid.at_fractional([site.frac])[0])
            assert sampled == pytest.approx(r.bvs_listed, abs=0.20), r.label

    def test_the_level_set_passes_through_the_real_sites(self, mapped):
        """The physically meaningful check: the surface where a Bi(III) probe
        would be correctly bonded should pass through where the bismuth
        actually is."""
        structure, results, grid = mapped
        vertices, _, _ = volume.isosurface(grid, 3.0)
        assert len(vertices) > 0
        for r in results:
            site = structure.sites[r.site_index]
            cart = structure.cell.to_cartesian(site.frac)
            assert float(np.linalg.norm(vertices - cart, axis=1).min()) < 0.35

    def test_most_of_the_cell_is_nowhere_near_the_level(self, mapped):
        """If the surface were not selective it would say nothing."""
        structure, results, grid = mapped
        rng = np.random.default_rng(0)
        sampled = grid.at_fractional(rng.random((300, 3)))
        near = np.abs(sampled - 3.0) < 0.3
        assert near.mean() < 0.25

    def test_the_energy_landscape_is_lowest_where_the_valence_is_right(
            self, mapped):
        structure, results, grid = mapped
        energy = volume.bond_valence_energy(grid, 3)
        index = np.unravel_index(np.argmin(energy.values), energy.shape)
        assert grid.values[index] == pytest.approx(3.0, abs=0.1)

    def test_the_energy_landscape_says_it_is_not_an_energy(self, mapped):
        structure, results, grid = mapped
        energy = volume.bond_valence_energy(grid, 3)
        assert any("not a calculated" in n for n in energy.notes)
        assert "v.u." in energy.units

    def test_a_structure_with_no_anions_is_refused(self):
        from facet.core.structure import Atom, Cell, Site, Structure

        cell = cell_from_parameters(4, 4, 4, 90, 90, 90)
        s = Structure(name="metal", cell=cell,
                      sites=[Site("Bi1", "Bi", [0, 0, 0])],
                      atoms=[Atom("Bi", np.zeros(3), np.zeros(3), 0, "Bi1")])
        s.sites[0].ox = 3
        with pytest.raises(ValueError):
            volume.bond_valence_grid(s, "Bi", 3, resolution=1.0)


# --- volumetric file readers -------------------------------------------------

def test_a_cube_file_round_trips_its_grid(tmp_path):
    """Written by hand, since a CUBE is short enough to be unambiguous."""
    n = 4
    values = np.arange(n ** 3, dtype=float)
    lines = ["comment", "another comment",
             f"    1    0.000000    0.000000    0.000000"]
    step = 2.0 / 0.529177210903          # 2 A in bohr
    for axis in range(3):
        vector = [0.0, 0.0, 0.0]
        vector[axis] = step
        lines.append(f"    {n} {vector[0]:.6f} {vector[1]:.6f} {vector[2]:.6f}")
    lines.append("    8    8.000000    0.000000    0.000000    0.000000")
    for start in range(0, len(values), 6):
        lines.append(" ".join(f"{v:.6e}" for v in values[start:start + 6]))
    (tmp_path / "f.cube").write_text("\n".join(lines) + "\n", encoding="utf-8")

    grid, structure = volume.read_cube(tmp_path / "f.cube")
    assert grid.shape == (n, n, n)
    assert np.allclose(grid.values.ravel(), values)
    assert grid.cell.a == pytest.approx(8.0, rel=1e-6)
    assert structure is not None and structure.n_atoms == 1


def test_an_xsf_drops_the_duplicated_plane(tmp_path):
    n = 5
    values = np.zeros(n ** 3)
    lines = ["BEGIN_BLOCK_DATAGRID_3D", " field",
             " BEGIN_DATAGRID_3D_test",
             f"   {n} {n} {n}",
             "   0.0 0.0 0.0",
             "   5.0 0.0 0.0", "   0.0 5.0 0.0", "   0.0 0.0 5.0"]
    for start in range(0, len(values), 6):
        lines.append(" ".join(f"{v:.5f}" for v in values[start:start + 6]))
    lines += [" END_DATAGRID_3D", "END_BLOCK_DATAGRID_3D"]
    (tmp_path / "f.xsf").write_text("\n".join(lines) + "\n", encoding="utf-8")

    grid, structure = volume.read_xsf(tmp_path / "f.xsf")
    assert grid.shape == (n - 1, n - 1, n - 1)
    assert any("duplicate" in note for note in grid.notes)


def test_an_unknown_volumetric_format_is_refused(tmp_path):
    (tmp_path / "x.dat").write_text("nothing", encoding="utf-8")
    with pytest.raises(ValueError) as raised:
        volume.read_volume(tmp_path / "x.dat")
    assert "CHGCAR" in str(raised.value)
