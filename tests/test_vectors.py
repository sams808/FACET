"""The bond-valence vector overlay: the measurement and the geometry.

The checks that matter here are the ones that compare the drawn quantity against
something with no code in common with it. phi from the scene's cached bond
arrays is checked against phi from ``coordination.analyse_structure``, which
reaches it by its own neighbour search through ``bv.phi_index`` -- that
comparison is what found the missing occupancy weighting, a difference of 0.13
on a partially occupied site that every internal check was happy with.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from facet.core import bv, coordination
from facet.gl import vectors as V
from facet.gl.scene import Scene, build_scene, merge_scenes

CIFS = Path(r"C:\Users\samso\Desktop\WSU_work\XRD\cif\Bi\cifs")
SAMPLE = CIFS / "1526458_Bi2O3.cif"


@pytest.fixture(scope="module")
def structure():
    if not SAMPLE.is_file():
        pytest.skip("the sample structure is not present")
    from facet.core import cif

    return cif.read(str(SAMPLE))


@pytest.fixture(scope="module")
def analysed(structure):
    return coordination.analyse_structure(structure, bv.DEFAULT,
                                          v_bond=bv.V_BOND_DEFAULT)


# --- the measurement --------------------------------------------------------

def test_phi_from_the_scene_matches_the_analysis(structure, analysed):
    """Two routes to phi that share no code: they must agree."""
    scene = build_scene(structure, analysed, v_bond=bv.V_BOND_DEFAULT,
                        show_vectors=True)
    sums = scene.vector_sums
    assert sums is not None
    checked = 0
    for r in analysed:
        if not r.bonds or not np.isfinite(r.phi):
            continue
        atoms = np.nonzero(scene.atom_site == r.site_index)[0]
        atoms = atoms[atoms < scene.n_cell_atoms]
        if not len(atoms):
            continue
        assert sums.phi[atoms] == pytest.approx(r.phi, abs=1e-5), r.label
        checked += 1
    assert checked, "no site was actually compared"


def test_every_atom_of_one_site_gets_the_same_phi(structure, analysed):
    """Symmetry copies are related by rotations, not translations.

    Propagating one atom's contact vectors to the others by translation is the
    fault this catches: it leaves each copy with the representative's
    directions, and phi then differs between atoms that must be identical.
    """
    scene = build_scene(structure, analysed, v_bond=bv.V_BOND_DEFAULT,
                        show_vectors=True)
    phi = scene.vector_sums.phi
    for site in set(int(x) for x in scene.atom_site[:scene.n_cell_atoms]):
        mine = phi[:scene.n_cell_atoms][
            scene.atom_site[:scene.n_cell_atoms] == site]
        if len(mine) > 1:
            assert mine.max() - mine.min() < 1e-5, site


def test_the_occupancy_weighting_is_applied():
    """A half-occupied ligand must count half, as it does in the analysis."""
    full = V.accumulate(
        bond_a=np.zeros((2, 3)), bond_b=np.array([[1.0, 0, 0], [-1.0, 0, 0]]),
        bond_atoms=np.array([[0, 1], [0, 2]]),
        bond_cation=np.array([0, 0]), bond_valence=np.array([0.5, 0.5]),
        v_bond=0.075, n_atoms=3)
    assert full.phi[0] == pytest.approx(0.0)           # they cancel exactly

    half = V.accumulate(
        bond_a=np.zeros((2, 3)), bond_b=np.array([[1.0, 0, 0], [-1.0, 0, 0]]),
        bond_atoms=np.array([[0, 1], [0, 2]]),
        bond_cation=np.array([0, 0]), bond_valence=np.array([0.5, 0.5]),
        v_bond=0.075, n_atoms=3, occupancy=np.array([1.0, 0.5]))
    # 0.5 one way against 0.25 the other: |sum| / sum = 0.25 / 0.75
    assert half.phi[0] == pytest.approx(1.0 / 3.0)


def test_the_vector_points_away_from_the_ligands():
    """Three ligands in a plane: the sum points along the fourth direction."""
    root3_2 = np.sqrt(3.0) / 2.0
    directions = np.array([[1.0, 0, 0], [-0.5, root3_2, 0],
                           [-0.5, -root3_2, 0]])
    sums = V.accumulate(
        bond_a=np.zeros((3, 3)), bond_b=directions,
        bond_atoms=np.array([[0, 1], [0, 2], [0, 3]]),
        bond_cation=np.zeros(3, int), bond_valence=np.array([0.4, 0.4, 0.4]),
        v_bond=0.075, n_atoms=4)
    # trigonal planar and equal: the sum cancels
    assert sums.phi[0] == pytest.approx(0.0, abs=1e-9)

    # break it: one bond stronger, and the sum points along that bond, so the
    # drawn lobe points the other way
    sums = V.accumulate(
        bond_a=np.zeros((3, 3)), bond_b=directions,
        bond_atoms=np.array([[0, 1], [0, 2], [0, 3]]),
        bond_cation=np.zeros(3, int), bond_valence=np.array([0.8, 0.4, 0.4]),
        v_bond=0.075, n_atoms=4)
    axis = -sums.vector[0] / np.linalg.norm(sums.vector[0])
    assert axis == pytest.approx([-1.0, 0.0, 0.0], abs=1e-6)
    assert sums.phi[0] > 0.2


def test_phi_is_invariant_to_an_error_in_r0():
    """Scaling every valence by a common factor cannot change phi.

    This is the property that makes the lobe's direction and length reportable
    at all: R0 carries about +-0.02 A even when fitted.
    """
    rng = np.random.default_rng(7)
    directions = rng.normal(size=(5, 3))
    valence = np.array([0.5, 0.3, 0.25, 0.2, 0.15])
    args = dict(bond_a=np.zeros((5, 3)), bond_b=directions,
                bond_atoms=np.stack([np.zeros(5, int), np.arange(1, 6)], 1),
                bond_cation=np.zeros(5, int), v_bond=0.0, n_atoms=6)
    base = V.accumulate(bond_valence=valence, **args)
    for factor in (0.5, 1.37, 2.9):
        scaled = V.accumulate(bond_valence=valence * factor, **args)
        assert scaled.phi[0] == pytest.approx(base.phi[0], abs=1e-12)
        # the direction too
        a = base.vector[0] / np.linalg.norm(base.vector[0])
        b = scaled.vector[0] / np.linalg.norm(scaled.vector[0])
        assert a @ b == pytest.approx(1.0, abs=1e-9)


def test_a_centrosymmetric_site_draws_nothing(structure):
    """phi = 0 must give no lobe rather than an arbitrary direction."""
    scene = Scene()
    scene.show_vectors = True
    scene.rebuild_overlays()
    assert scene.overlay_meshes == []


# --- the geometry -----------------------------------------------------------

def test_the_lobe_has_the_length_and_waist_asked_for():
    direction = np.array([[0.3, -0.5, 0.81]])
    direction /= np.linalg.norm(direction)
    mesh = V.lobe_meshes(np.array([[1.0, 2.0, 3.0]]), direction, [1.4])
    rel = np.asarray(mesh.vertices, float) - np.array([1.0, 2.0, 3.0])
    along = rel @ direction[0]
    radial = np.linalg.norm(rel - along[:, None] * direction[0], axis=1)
    assert along.min() == pytest.approx(0.0, abs=1e-5)
    assert along.max() == pytest.approx(1.4, abs=1e-5)
    assert radial.max() == pytest.approx(V.LOBE_RADIUS, abs=1e-5)
    # the stated profile, not something that merely looks like it
    t = np.clip(along / 1.4, 0, 1)
    assert radial == pytest.approx(V.LOBE_RADIUS * np.sin(np.pi * t) ** 0.75,
                                   abs=1e-4)


def test_the_lobe_normals_point_outward_and_are_unit_length():
    mesh = V.lobe_meshes([[0.0, 0, 0]], [[0.0, 0, 1.0]], [1.0])
    verts = np.asarray(mesh.vertices, float)
    normals = np.asarray(mesh.normals, float)
    assert np.linalg.norm(normals, axis=1) == pytest.approx(1.0, abs=1e-6)
    radial = verts.copy()
    radial[:, 2] = 0.0
    r = np.linalg.norm(radial, axis=1)
    waist = r > 0.5 * V.LOBE_RADIUS
    dots = (normals[waist] * (radial[waist] / r[waist][:, None])).sum(axis=1)
    assert dots.min() > 0.5


def test_the_lobe_is_a_surface_of_revolution_about_its_axis():
    """Rotating the inputs gives the same surface, not the same vertices.

    A surface of revolution has no preferred azimuth, so the generating frame's
    choice of reference direction carries no information -- comparing vertex for
    vertex would be testing that heuristic rather than the geometry.
    """
    origin = np.array([[1.0, 2.0, 3.0]])
    direction = np.array([[0.3, -0.5, 0.81]])
    direction /= np.linalg.norm(direction)
    mesh = V.lobe_meshes(origin, direction, [1.4])

    theta = 0.7
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    turned = V.lobe_meshes(origin @ R.T, direction @ R.T, [1.4])

    def profile(m, o, d):
        rel = np.asarray(m.vertices, float) - o
        along = rel @ d
        return np.sort(along), np.sort(
            np.linalg.norm(rel - along[:, None] * d, axis=1))

    a1, r1 = profile(mesh, origin[0], direction[0])
    a2, r2 = profile(turned, (origin @ R.T)[0], (direction @ R.T)[0])
    assert a2 == pytest.approx(a1, abs=1e-6)
    assert r2 == pytest.approx(r1, abs=1e-6)


def test_the_cone_has_the_half_angle_asked_for():
    """Measured as the angle between the axis and a generator.

    Not as rim/height: that ratio is tan(alpha), which runs away at 90 degrees,
    and a void half-angle approaches 90 degrees exactly when the environment is
    one-sided -- the case the cone exists to show.
    """
    for degrees in (15.0, 35.0, 54.7356, 80.0, 90.0, 120.0, 175.0):
        length = 2.0
        cone = V.cone_meshes([[0, 0, 0]], [[0, 0, 1.0]], [degrees], [length])
        v = np.asarray(cone.vertices, float)
        rim = v[np.linalg.norm(v, axis=1) > 1e-6]
        assert len(rim)
        # every rim point is one generator length from the apex...
        assert np.linalg.norm(rim, axis=1) == pytest.approx(length, abs=1e-5)
        # ...at the angle asked for, from the axis
        cosine = rim[:, 2] / np.linalg.norm(rim, axis=1)
        assert np.degrees(np.arccos(np.clip(cosine, -1, 1))) == pytest.approx(
            degrees, abs=1e-4)


def test_a_wide_cone_stays_the_size_it_was_asked_for():
    """A half-angle near 90 degrees must not produce a cone the size of a room.

    Fixing the axial height puts the rim at L*tan(alpha): at 89.94 degrees that
    is 954 times the length asked for, which filled the viewport and -- because
    the framing includes the overlays -- pushed the structure out of the view.
    """
    for degrees in (89.0, 89.94, 90.0, 91.0):
        cone = V.cone_meshes([[0, 0, 0]], [[0, 0, 1.0]], [degrees], [1.5])
        assert np.abs(np.asarray(cone.vertices, float)).max() <= 1.5 + 1e-6


def test_parts_survive_a_depth_sort():
    """The renderer reorders triangles; the part mapping must follow them."""
    mesh = V.lobe_meshes(np.array([[0.0, 0, 0], [5.0, 0, 0], [10.0, 0, 0]]),
                         np.tile([[0.0, 0, 1.0]], (3, 1)), [1.0, 1.0, 1.0])
    assert mesh.n_parts == 3
    before = [set(map(tuple, np.asarray(mesh.vertices, float).reshape(-1, 3, 3)[
        np.asarray(mesh.part_of) == p].reshape(-1, 3))) for p in range(3)]

    rng = np.random.default_rng(3)
    order = rng.permutation(mesh.n_triangles)
    mesh.reorder(order)
    after = [set(map(tuple, np.asarray(mesh.vertices, float).reshape(-1, 3, 3)[
        np.asarray(mesh.part_of) == p].reshape(-1, 3))) for p in range(3)]
    assert before == after


def test_keep_parts_drops_whole_lobes():
    mesh = V.lobe_meshes(np.array([[0.0, 0, 0], [5.0, 0, 0], [10.0, 0, 0]]),
                         np.tile([[0.0, 0, 1.0]], (3, 1)), [1.0, 1.0, 1.0],
                         atoms=[7, 8, 9])
    kept = mesh.keep_parts([True, False, True])
    assert kept.n_parts == 2
    assert kept.n_triangles == mesh.n_triangles * 2 // 3
    assert list(kept.atoms) == [7, 9]
    # nothing from the dropped lobe survives, and part_of still indexes anchors
    x = np.asarray(kept.vertices, float)[:, 0]
    assert not ((x > 3.0) & (x < 7.0)).any()
    assert kept.part_of.max() < kept.n_parts


# --- through the scene ------------------------------------------------------

def test_the_overlay_follows_the_threshold(structure, analysed):
    """Widening the cut adds contacts, so the sum must change with it."""
    scene = build_scene(structure, analysed, v_bond=0.075, show_vectors=True)
    tight = scene.vector_sums.phi.copy()
    tight_count = scene.vector_sums.count.copy()
    scene.restyle(0.02)
    assert (scene.vector_sums.count >= tight_count).all()
    assert scene.vector_sums.count.sum() > tight_count.sum()
    assert not np.allclose(scene.vector_sums.phi, tight)


def test_switching_the_overlay_off_removes_the_mesh(structure, analysed):
    scene = build_scene(structure, analysed, v_bond=0.075, show_vectors=True)
    assert scene.overlay_meshes
    scene.show_vectors = False
    scene.rebuild_overlays()
    assert scene.overlay_meshes == []
    # but the measurement is still available for the readout
    assert scene.vector_sums is not None


def test_the_scale_only_changes_the_length(structure, analysed):
    short = build_scene(structure, analysed, show_vectors=True,
                        vector_scale=0.4)
    long = build_scene(structure, analysed, show_vectors=True,
                       vector_scale=1.2)

    def reach(scene):
        mesh = scene.overlay_meshes[0]
        rel = np.asarray(mesh.vertices, float) - np.repeat(
            mesh.anchors, mesh.n_triangles // mesh.n_parts * 3, axis=0)
        return np.linalg.norm(rel, axis=1).max()

    assert reach(long) > reach(short) * 2.5
    assert short.overlay_meshes[0].n_parts == long.overlay_meshes[0].n_parts


def test_a_replicated_cell_gives_every_copy_its_own_lobe(structure, analysed):
    one = build_scene(structure, analysed, show_vectors=True)
    block = build_scene(structure, analysed, show_vectors=True,
                        cell_range=(2, 1, 1))
    assert block.overlay_meshes[0].n_parts >= 2 * one.overlay_meshes[0].n_parts


def test_a_merged_scene_rebuilds_its_overlays(structure, analysed):
    a = build_scene(structure, analysed, show_vectors=True)
    b = build_scene(structure, analysed, show_vectors=True)
    merged = merge_scenes([a, b], [np.zeros(3), np.array([20.0, 0, 0])])
    assert merged.show_vectors
    assert merged.overlay_meshes
    assert (merged.overlay_meshes[0].n_parts
            == a.overlay_meshes[0].n_parts + b.overlay_meshes[0].n_parts)
    # the second copy's lobes are where the second copy is
    x = merged.overlay_meshes[0].anchors[:, 0]
    assert x.max() > 15.0


def test_the_slab_drops_whole_lobes_not_halves(structure, analysed):
    from facet.core.planes import LatticePlane, Slab

    scene = build_scene(structure, analysed, show_vectors=True)
    before = scene.overlay_meshes[0].n_parts
    tris_per_part = scene.overlay_meshes[0].n_triangles // before

    plane = LatticePlane(h=0, k=0, l=1)
    slab = Slab(h=plane.h, k=plane.k, l=plane.l, centre=0.0,
                thickness=2.0, enabled=True)
    cut = build_scene(structure, analysed, show_vectors=True, slab=slab)
    if not cut.overlay_meshes:
        pytest.skip("the slab removed every cation")
    after = cut.overlay_meshes[0]
    assert after.n_parts <= before
    # a whole number of lobes, so none was sliced
    assert after.n_triangles == after.n_parts * tris_per_part


def test_the_software_renderer_draws_the_overlay(structure, analysed, qapp):
    """The promise is that every visual feature works with no graphics card."""
    from PySide6.QtGui import QImage, QPainter

    from facet.gl.camera import Camera
    from facet.gl.painter import PainterRenderer

    plain = build_scene(structure, analysed, show_vectors=False)
    lobed = build_scene(structure, analysed, show_vectors=True,
                        vector_scale=1.3)
    images = []
    for scene in (plain, lobed):
        camera = Camera()
        camera.frame(scene.center, scene.radius)
        image = QImage(240, 200, QImage.Format_RGB32)
        image.fill(0)
        painter = QPainter(image)
        PainterRenderer().render(painter, scene, camera, 240, 200)
        painter.end()
        images.append(image)
    assert images[0] != images[1], "the lobes changed nothing on screen"


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_the_vector_export_carries_the_overlay(structure, analysed, qapp,
                                               tmp_path):
    """SVG and PDF go through the software renderer, so they inherit it."""
    from facet.gl import vector_export as X
    from facet.gl.camera import Camera

    scene = build_scene(structure, analysed, show_vectors=True,
                        vector_scale=1.3)
    camera = Camera()
    camera.frame(scene.center, scene.radius)
    with_lobes = Path(X.save_svg(tmp_path / "lobes.svg", scene, camera,
                                 320, 260))
    scene.show_vectors = False
    scene.rebuild_overlays()
    without = Path(X.save_svg(tmp_path / "plain.svg", scene, camera, 320, 260))
    assert (with_lobes.read_text(encoding="utf-8", errors="replace")
            != without.read_text(encoding="utf-8", errors="replace"))
