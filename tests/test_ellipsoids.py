"""Displacement ellipsoids, from the tensor to the pixels.

The arithmetic is checked in tests/test_adp.py. What is checked here is that the
right shape reaches the right atom and is drawn where it belongs -- which is a
different kind of fault, and a quieter one: an ellipsoid drawn with another
atom's tensor is a plausible ellipsoid, and looks like a measurement.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from facet.core import adp, cif
from facet.gl.scene import Style, build_scene, merge_scenes

SAMPLE = Path(r"C:\Users\samso\Desktop\WSU_work\XRD\cif"
              r"\1004091_BiNa3O8P2.cif")
NPD = Path(r"C:\Users\samso\Desktop\WSU_work\XRD\cif\7023720_BiPO4.cif")

pytestmark = pytest.mark.skipif(not SAMPLE.is_file(),
                                reason="the reference structure is not present")


@pytest.fixture(scope="module")
def structure():
    return cif.read(SAMPLE)


def _semi_axes(matrix):
    """The three semi-axis lengths of a shape matrix, ascending."""
    return np.sort(np.linalg.norm(np.asarray(matrix, float), axis=0))


def _expected(structure, site_index, probability=0.50):
    site = structure.sites[site_index]
    shape = adp.for_site(structure.cell, site)
    if shape is not None and shape.is_ellipsoid:
        return np.sort(shape.radii(probability)), True
    if site.u_iso:
        r = float(np.sqrt(max(site.u_iso, 0.0))) * adp.scale_for(probability)
        return np.array([r, r, r]), False
    return None, False


# ---------------------------------------------------------------------------
# the shape reaches the atom
# ---------------------------------------------------------------------------

def test_every_atom_carries_its_own_site_s_ellipsoid(structure):
    scene = build_scene(structure, style=Style.ELLIPSOIDS)
    assert scene.atom_shape is not None
    assert len(scene.atom_shape) == scene.n_atoms
    for i in range(scene.n_atoms):
        want, anisotropic = _expected(structure, int(scene.atom_site[i]))
        assert want is not None
        assert _semi_axes(scene.atom_shape[i]) == pytest.approx(want, abs=1e-6)
        assert bool(scene.atom_anisotropic[i]) is anisotropic


@pytest.mark.parametrize("cell_range", [(1, 1, 1), (2, 1, 1), (2, 2, 2)])
def test_the_shape_survives_the_cell_repeats(structure, cell_range):
    """A lattice translation moves an atom and not its tensor.

    The atom arrays are rebuilt by the repeat; if the shapes were not tiled
    with them the array lengths would disagree, and if they were transformed
    the repeated atoms would be drawn with the wrong orientation.
    """
    scene = build_scene(structure, style=Style.ELLIPSOIDS,
                        cell_range=cell_range)
    assert len(scene.atom_shape) == scene.n_atoms
    assert len(scene.atom_anisotropic) == scene.n_atoms
    for i in range(scene.n_atoms):
        want, _ = _expected(structure, int(scene.atom_site[i]))
        assert _semi_axes(scene.atom_shape[i]) == pytest.approx(want, abs=1e-6)


def test_the_shape_survives_a_slab(structure):
    """A slab re-indexes every atom array. One left behind misaligns them all."""
    from facet.core import planes as planes_mod

    slab = planes_mod.Slab(h=0, k=0, l=1, centre=0.0, thickness=5.0,
                           enabled=True)
    scene = build_scene(structure, style=Style.ELLIPSOIDS, slab=slab,
                        cell_range=(1, 1, 1))
    full = build_scene(structure, style=Style.ELLIPSOIDS, cell_range=(1, 1, 1))
    assert 0 < scene.n_atoms < full.n_atoms, (
        "the slab removed nothing, so this proves nothing")
    assert len(scene.atom_shape) == scene.n_atoms
    for i in range(scene.n_atoms):
        want, _ = _expected(structure, int(scene.atom_site[i]))
        assert _semi_axes(scene.atom_shape[i]) == pytest.approx(want, abs=1e-6)


def test_the_shape_survives_the_merge_of_two_structures(structure):
    """The overlay concatenates site indices without offsetting them.

    So the shapes cannot be looked up through atom_site after a merge, which is
    the reason they are carried per atom. Each half of the merged scene must
    keep the shapes its own scene had.
    """
    one = build_scene(structure, style=Style.ELLIPSOIDS)
    two = build_scene(structure, style=Style.ELLIPSOIDS)
    merged = merge_scenes([one, two], [np.zeros(3), np.array([40.0, 0, 0])])

    assert merged.atom_shape is not None
    assert len(merged.atom_shape) == merged.n_atoms == one.n_atoms + two.n_atoms
    assert merged.atom_shape[:one.n_atoms] == pytest.approx(one.atom_shape)
    assert merged.atom_shape[one.n_atoms:] == pytest.approx(two.atom_shape)


def test_a_scene_without_ellipsoids_has_no_shapes(structure):
    for style in (Style.BALL_AND_STICK, Style.SPACE_FILLING, Style.STICK):
        scene = build_scene(structure, style=style)
        assert scene.atom_shape is None
        assert scene.atom_anisotropic is None


# ---------------------------------------------------------------------------
# what the shape is
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("probability", [0.10, 0.50, 0.90, 0.99])
def test_the_probability_scales_every_ellipsoid(structure, probability):
    base = build_scene(structure, style=Style.ELLIPSOIDS,
                       ellipsoid_probability=0.50)
    other = build_scene(structure, style=Style.ELLIPSOIDS,
                        ellipsoid_probability=probability)
    ratio = adp.scale_for(probability) / adp.scale_for(0.50)
    assert other.atom_shape == pytest.approx(
        np.asarray(base.atom_shape) * ratio, rel=1e-5)


def test_a_site_with_no_tensor_is_a_sphere_and_says_so():
    """Not every file refines anisotropically, and the two must be tellable apart.

    A sphere built from U_iso is a real measurement, but it has no principal
    axes -- so it is drawn without the octant boundaries that assert them.
    """
    if not NPD.is_file():
        pytest.skip("the reference structure is not present")
    structure = cif.read(NPD)
    scene = build_scene(structure, style=Style.ELLIPSOIDS)

    by_label = {}
    for i in range(scene.n_atoms):
        by_label.setdefault(structure.sites[int(scene.atom_site[i])].label, i)

    # P1's tensor is not positive definite, so it cannot be drawn as one
    i = by_label["P1"]
    axes = _semi_axes(scene.atom_shape[i])
    assert axes == pytest.approx(axes[0], rel=1e-9)        # a sphere
    assert not scene.atom_anisotropic[i]

    # and a site that does have one is marked, and is not a sphere
    j = by_label["O1"]
    assert scene.atom_anisotropic[j]
    assert _semi_axes(scene.atom_shape[j])[2] > \
        _semi_axes(scene.atom_shape[j])[0] * 2


def test_bonds_are_drawn_thin_beside_ellipsoids(structure):
    """A 50% ellipsoid is a tenth of an angstrom; an ordinary bond is wider.

    At full width the bonds are the picture and the ellipsoids are specks
    hanging off them, which is why every ellipsoid plot since ORTEP has drawn
    thin sticks.
    """
    plain = build_scene(structure, style=Style.BALL_AND_STICK)
    ellipsoids = build_scene(structure, style=Style.ELLIPSOIDS)
    assert plain.n_bonds == ellipsoids.n_bonds > 0
    assert float(np.max(ellipsoids.bond_radius)) < float(np.max(plain.bond_radius))

    largest = float(np.max(np.linalg.norm(ellipsoids.atom_shape, axis=1)))
    assert float(np.max(ellipsoids.bond_radius)) < largest


def test_the_drawn_radius_matches_the_drawn_shape(structure):
    """Everything that measures an atom on screen reads atom_radius.

    The label offset, the selection ring, the software tier's culling and its
    picking all do. In this style the drawn size is the ellipsoid, so the
    radius has to be its largest semi-axis rather than an atomic radius that
    has nothing to do with what is on screen.
    """
    scene = build_scene(structure, style=Style.ELLIPSOIDS)
    largest = np.linalg.norm(np.asarray(scene.atom_shape), axis=1).max(axis=1)
    assert scene.atom_radius == pytest.approx(largest, rel=1e-5)


# ---------------------------------------------------------------------------
# what is drawn
# ---------------------------------------------------------------------------

def test_the_software_tier_draws_the_projected_ellipse(structure, qapp):
    """The drawn outline against the projected surface, computed separately.

    Points sampled on the ellipsoid in world coordinates and pushed through the
    camera give the outline directly, with none of the conic arithmetic the
    renderer uses.
    """
    from facet.gl.camera import Camera
    from facet.gl.painter import PainterRenderer

    scene = build_scene(structure, style=Style.ELLIPSOIDS)
    camera = Camera()
    camera.frame(scene.center, scene.radius)
    width, height = 900, 700

    renderer = PainterRenderer()
    ellipses = renderer._project_ellipsoids(scene, camera, width, height)
    assert ellipses is not None and len(ellipses) == scene.n_atoms

    rng = np.random.default_rng(0)
    unit = rng.normal(size=(4000, 3))
    unit /= np.linalg.norm(unit, axis=1)[:, None]

    for i in range(0, scene.n_atoms, 11):
        matrix = np.asarray(scene.atom_shape[i], float)
        surface = scene.atom_position[i] + unit @ matrix.T
        flat = camera.project(surface, width, height)[:, :2]
        centre = camera.project(scene.atom_position[i:i + 1],
                                width, height)[0, :2]
        d = flat - centre
        # <xx> over a projected shell is a^2 / 3
        values = np.sqrt(np.clip(
            np.linalg.eigvalsh((d.T @ d) / len(d) * 3.0), 0, None))
        major, minor, _ = ellipses[i]
        assert values[1] == pytest.approx(major, rel=0.05)
        assert values[0] == pytest.approx(minor, rel=0.06)


def test_an_isotropic_tensor_draws_the_circle_a_sphere_would(structure, qapp):
    from facet.gl.camera import Camera
    from facet.gl.painter import PainterRenderer

    scene = build_scene(structure, style=Style.ELLIPSOIDS)
    scene.atom_shape = np.tile(np.eye(3, dtype=np.float32) * 0.5,
                               (scene.n_atoms, 1, 1))
    camera = Camera()
    camera.frame(scene.center, scene.radius)
    camera.orthographic = True             # exact, with no perspective spread

    renderer = PainterRenderer()
    renderer._scale = (700 * 0.5) / max(camera.half_height(), 1e-6)
    renderer._distance = max(camera.distance, 1e-6)
    renderer._perspective = False
    ellipses = renderer._project_ellipsoids(scene, camera, 900, 700)

    assert ellipses[:, 0] == pytest.approx(ellipses[:, 1], rel=1e-6)
    expected = renderer.radius_at(0.5, -camera.distance)
    assert ellipses[:, 0] == pytest.approx(expected, rel=1e-6)


def test_the_vertex_buffer_carries_the_inverse_shape(structure):
    """The shader needs M^-1, and a singular M has none.

    A zero tensor can only come from a file that wrote one, and the honest
    drawing of it is a small sphere rather than an inverse that would paint the
    whole screen.
    """
    from facet.gl import buffers

    scene = build_scene(structure, style=Style.ELLIPSOIDS)
    arrays = buffers.ellipsoid_vertices(scene)
    rows = np.stack([arrays["aShapeX"], arrays["aShapeY"], arrays["aShapeZ"]],
                    axis=1)
    per_atom = rows[::buffers.VERTS_PER_SPHERE]
    product = per_atom @ np.asarray(scene.atom_shape)
    assert product == pytest.approx(
        np.tile(np.eye(3), (scene.n_atoms, 1, 1)), abs=1e-4)

    scene.atom_shape = np.zeros_like(scene.atom_shape)
    arrays = buffers.ellipsoid_vertices(scene)
    assert np.all(np.isfinite(arrays["aShapeX"]))
