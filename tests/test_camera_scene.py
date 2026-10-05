"""Camera matrices and scene construction.

Both are pure numpy, so they are tested without a window. The camera is the
part most likely to be subtly wrong in a way that only shows up as "the model
rotates oddly", which is hard to notice and harder to bisect.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from conftest import sample_cif

from facet.core import bv
from facet.gl import camera as cam
from facet.gl.scene import Scene, Style, bond_radius_for

SAMPLE = sample_cif("1526458", "1526458_Bi2O3.cif")


# --- quaternions -------------------------------------------------------------

def test_identity_quaternion_is_identity_matrix():
    assert np.allclose(cam.quat_to_matrix(cam.quat_identity()), np.eye(3))


def test_quaternion_multiplication_composes_rotations():
    qx = cam.quat_from_axis_angle([1, 0, 0], math.pi / 2)
    composed = cam.quat_to_matrix(cam.quat_multiply(qx, qx))
    expected = cam.quat_to_matrix(cam.quat_from_axis_angle([1, 0, 0], math.pi))
    assert np.allclose(composed, expected, atol=1e-12)


def test_rotation_matrices_are_orthonormal_with_unit_determinant():
    rng = np.random.default_rng(3)
    for _ in range(20):
        q = cam.quat_from_axis_angle(rng.normal(size=3), rng.uniform(-math.pi, math.pi))
        m = cam.quat_to_matrix(q)
        assert np.allclose(m @ m.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(m) == pytest.approx(1.0, abs=1e-12)


def test_quarter_turn_about_z_moves_x_to_y():
    m = cam.quat_to_matrix(cam.quat_from_axis_angle([0, 0, 1], math.pi / 2))
    assert np.allclose(m @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-12)


# --- projection --------------------------------------------------------------

def test_perspective_maps_the_near_and_far_planes_to_minus_one_and_one():
    p = cam.perspective(30.0, 1.5, 1.0, 100.0)
    for z, want in ((-1.0, -1.0), (-100.0, 1.0)):
        clip = p @ np.array([0.0, 0.0, z, 1.0])
        assert clip[2] / clip[3] == pytest.approx(want, abs=1e-9)


def test_orthographic_maps_the_half_height_to_the_top_edge():
    o = cam.orthographic(5.0, 2.0, -50.0, 50.0)
    clip = o @ np.array([0.0, 5.0, 0.0, 1.0])
    assert clip[1] / clip[3] == pytest.approx(1.0, abs=1e-12)
    clip = o @ np.array([10.0, 0.0, 0.0, 1.0])     # half width = 5 * 2
    assert clip[0] / clip[3] == pytest.approx(1.0, abs=1e-12)


def test_look_at_puts_the_target_at_the_origin_of_view_space():
    m = cam.look_at([3, 4, 5], [1, 1, 1], [0, 1, 0])
    out = m @ np.array([1.0, 1.0, 1.0, 1.0])
    assert np.allclose(out[:3], [0, 0, -np.linalg.norm([2, 3, 4])], atol=1e-9)


def test_look_at_survives_looking_straight_up():
    """Degenerate up vector must not produce NaNs."""
    m = cam.look_at([0, 5, 0], [0, 0, 0], [0, 1, 0])
    assert np.all(np.isfinite(m))


# --- the camera --------------------------------------------------------------

def test_framing_fits_the_bounding_sphere_in_view():
    """The sphere's silhouette should just fill the viewport.

    Note it is the silhouette, not the pole: the tangent point of a sphere sits
    nearer the camera than its extreme world coordinate, so the point (0, R, 0)
    projects comfortably inside the edge. Testing the pole would be testing the
    wrong geometry.
    """
    c = cam.Camera(fov=30.0)
    c.frame([0, 0, 0], 5.0, margin=1.0)
    ring = [[0.0, 5.0 * math.cos(t), 5.0 * math.sin(t)]
            for t in np.linspace(0, 2 * math.pi, 256)]
    y = c.project(ring, 800, 800)[:, 1]
    assert y.min() == pytest.approx(0.0, abs=2.0)
    assert y.max() == pytest.approx(800.0, abs=2.0)


def test_framing_keeps_every_point_of_the_sphere_on_screen():
    c = cam.Camera(fov=30.0)
    c.frame([0, 0, 0], 5.0)                      # default margin
    rng = np.random.default_rng(1)
    v = rng.normal(size=(400, 3))
    pts = 5.0 * v / np.linalg.norm(v, axis=1)[:, None]
    px = c.project(pts, 800, 600)
    assert px[:, 0].min() >= 0 and px[:, 0].max() <= 800
    assert px[:, 1].min() >= 0 and px[:, 1].max() <= 600


def test_target_projects_to_the_centre_of_the_viewport():
    c = cam.Camera()
    c.frame([1.0, 2.0, 3.0], 4.0)
    px = c.project([[1.0, 2.0, 3.0]], 640, 480)[0]
    assert px[0] == pytest.approx(320.0, abs=1e-6)
    assert px[1] == pytest.approx(240.0, abs=1e-6)


def test_target_stays_centred_under_rotation():
    c = cam.Camera()
    c.frame([0, 0, 0], 3.0)
    c.drag_rotate(100, 100, 260, 190, 500, 500)
    px = c.project([[0.0, 0.0, 0.0]], 500, 500)[0]
    assert px[0] == pytest.approx(250.0, abs=1e-6)
    assert px[1] == pytest.approx(250.0, abs=1e-6)


def test_rotation_keeps_the_camera_at_the_same_distance():
    c = cam.Camera()
    c.frame([0, 0, 0], 3.0)
    before = np.linalg.norm(c.eye() - c.target)
    c.drag_rotate(10, 10, 300, 220, 400, 400)
    assert np.linalg.norm(c.eye() - c.target) == pytest.approx(before, abs=1e-9)


def test_dragging_right_and_back_returns_to_the_start():
    """An arcball must be path-reversible, or dragging feels like it drifts."""
    c = cam.Camera()
    c.frame([0, 0, 0], 3.0)
    start = c.orientation.copy()
    c.drag_rotate(200, 200, 300, 240, 500, 500)
    c.drag_rotate(300, 240, 200, 200, 500, 500)
    m0, m1 = cam.quat_to_matrix(start), cam.quat_to_matrix(c.orientation)
    assert np.allclose(m0, m1, atol=1e-9)


def test_panning_moves_the_scene_with_the_pointer():
    """The point under the pointer must keep up with it, pixel for pixel."""
    c = cam.Camera()
    c.frame([0, 0, 0], 3.0)
    before = c.project([[0.0, 0.0, 0.0]], 400, 400)[0]
    c.drag_pan(40.0, 0.0, 400, 400)
    after = c.project([[0.0, 0.0, 0.0]], 400, 400)[0]
    assert after[0] - before[0] == pytest.approx(40.0, abs=0.5)

    before = after
    c.drag_pan(0.0, -25.0, 400, 400)
    after = c.project([[0.0, 0.0, 0.0]], 400, 400)[0]
    assert after[1] - before[1] == pytest.approx(-25.0, abs=0.5)


def test_zoom_changes_distance_and_stays_bounded():
    c = cam.Camera()
    c.frame([0, 0, 0], 2.0)
    near = c.distance
    c.zoom(3)
    assert c.distance < near
    for _ in range(300):
        c.zoom(5)
    assert c.distance >= c.scene_radius * 0.05 - 1e-9
    for _ in range(300):
        c.zoom(-5)
    assert c.distance <= c.scene_radius * 60.0 + 1e-9


def test_changing_the_lens_does_not_change_the_framing():
    """set_fov is a lens control, not a zoom.

    The invariant is that pulling back to a longer lens leaves the object
    framed exactly as re-framing at that lens would -- so the distance must
    match what frame() would have chosen.
    """
    c = cam.Camera(fov=45.0)
    c.frame([0, 0, 0], 4.0)
    c.set_fov(12.0)

    reference = cam.Camera(fov=12.0)
    reference.frame([0, 0, 0], 4.0)
    assert c.distance == pytest.approx(reference.distance, rel=1e-12)

    # and the silhouette still fills the frame
    ring = [[0.0, 4.0 * math.cos(t), 4.0 * math.sin(t)]
            for t in np.linspace(0, 2 * math.pi, 128)]
    y = c.project(ring, 600, 600)[:, 1]
    assert y.min() >= -2.0 and y.max() <= 602.0


def test_lens_is_clamped_to_a_usable_range():
    c = cam.Camera()
    c.frame([0, 0, 0], 3.0)
    c.set_fov(0.001)
    assert c.fov == pytest.approx(cam.Camera.MIN_FOV)
    c.set_fov(500.0)
    assert c.fov == pytest.approx(cam.Camera.MAX_FOV)


def test_clip_planes_stay_positive_when_zoomed_inside_the_structure():
    c = cam.Camera()
    c.frame([0, 0, 0], 5.0)
    for _ in range(60):
        c.zoom(5)
    near, far = c.clip_planes()
    assert near > 0 and far > near


def test_view_along_an_axis_points_the_camera_down_it():
    c = cam.Camera()
    c.frame([0, 0, 0], 2.0)
    c.view_along([0, 0, 1])
    direction = c.target - c.eye()
    direction /= np.linalg.norm(direction)
    assert np.allclose(direction, [0, 0, -1], atol=1e-6)


def test_orthographic_projection_removes_perspective_divergence():
    """Two equal rods at different depths must measure the same on screen."""
    c = cam.Camera(orthographic=True)
    c.frame([0, 0, 0], 6.0)
    near = c.project([[0.0, -1.0, 2.0], [0.0, 1.0, 2.0]], 500, 500)
    far = c.project([[0.0, -1.0, -2.0], [0.0, 1.0, -2.0]], 500, 500)
    assert abs(near[1, 1] - near[0, 1]) == pytest.approx(
        abs(far[1, 1] - far[0, 1]), abs=1e-6)


# --- bond radius -------------------------------------------------------------

def test_bond_radius_increases_with_valence():
    v = np.array([0.02, 0.05, 0.1, 0.3, 0.6])
    r = bond_radius_for(v, 0.0)          # threshold below all, no pinching
    assert np.all(np.diff(r) > 0)


def test_sub_threshold_contacts_are_visibly_thinner():
    """The threshold must read as a step in thickness, not a gradient."""
    just_below = bond_radius_for(np.array([0.0749]), 0.075)[0]
    just_above = bond_radius_for(np.array([0.0751]), 0.075)[0]
    assert just_above > just_below * 1.8


def test_bond_radius_is_never_zero_or_negative():
    r = bond_radius_for(np.array([0.0, 1e-6, 0.001]), 0.075)
    assert np.all(r > 0)


# --- scene -------------------------------------------------------------------

@pytest.mark.skipif(not SAMPLE.exists(), reason="sample structure not present")
class TestSceneFromRealStructure:

    @pytest.fixture(scope="class")
    def scene(self):
        from facet.core import cif
        from facet.gl.scene import build_scene

        structure = cif.read(SAMPLE)
        return build_scene(structure, polyhedron_site=None)

    def test_every_atom_of_the_cell_is_present(self, scene):
        assert scene.n_cell_atoms == 20

    def test_periodic_images_that_close_a_bond_are_drawn(self, scene):
        """Without these, every bond crossing a cell boundary is drawn running
        out to an atom that is not on screen -- a spray of stubs into space."""
        assert scene.n_atoms > scene.n_cell_atoms

    def test_no_bond_ends_at_an_atom_that_is_not_drawn(self, scene):
        drawn = {(round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3))
                 for p in scene.atom_position}
        missing = [i for i in range(scene.n_bonds)
                   if (round(float(scene.bond_b[i][0]), 3),
                       round(float(scene.bond_b[i][1]), 3),
                       round(float(scene.bond_b[i][2]), 3)) not in drawn]
        assert not missing, f"{len(missing)} bonds end nowhere"

    def test_bonds_were_found_and_carry_their_valence(self, scene):
        assert scene.n_bonds > 0
        assert np.all(scene.bond_valence > 0)
        assert len(scene.bond_valence) == scene.n_bonds

    def test_sub_threshold_contacts_are_kept_not_dropped(self, scene):
        """The contested contacts must be in the scene, or the argument the
        application exists to make cannot be seen."""
        assert (scene.bond_valence < scene.v_bond).any(), \
            "no sub-threshold contacts retained"

    def test_restyle_changes_the_bond_count_without_rebuilding(self, scene):
        n_before = scene.n_bonds
        wide = scene.bonds_above_threshold()
        scene.restyle(0.30)
        assert scene.n_bonds == n_before          # nothing rebuilt
        assert scene.bonds_above_threshold() < wide
        scene.restyle(0.01)
        assert scene.bonds_above_threshold() > wide
        scene.restyle(bv.V_BOND_DEFAULT)
        assert scene.bonds_above_threshold() == wide

    def test_restyle_fades_sub_threshold_bonds(self, scene):
        scene.restyle(0.20)
        below = scene.bond_valence < 0.20
        assert below.any()
        base = scene._bond_base_a[below]
        faded = scene.bond_color_a[below]
        assert np.all(np.linalg.norm(faded - base, axis=1) > 1e-6)
        scene.restyle(bv.V_BOND_DEFAULT)

    def test_unit_cell_has_twelve_edges(self, scene):
        assert scene.cell_segments.shape == (12, 2, 3)

    def test_bounding_sphere_contains_every_atom(self, scene):
        d = np.linalg.norm(scene.atom_position - scene.center, axis=1)
        assert np.all(d <= scene.radius + 1e-6)

    def test_scene_arrays_agree_in_length(self, scene):
        n = scene.n_atoms
        assert len(scene.atom_radius) == n
        assert len(scene.atom_color) == n
        assert len(scene.atom_label) == n
        assert len(scene.atom_site) == n
        m = scene.n_bonds
        for arr in (scene.bond_a, scene.bond_b, scene.bond_radius,
                    scene.bond_color_a, scene.bond_color_b, scene.bond_atoms):
            assert len(arr) == m

    def test_arrays_are_float32_for_direct_upload(self, scene):
        for arr in (scene.atom_position, scene.atom_radius, scene.atom_color,
                    scene.bond_a, scene.bond_radius):
            assert arr.dtype == np.float32

    def test_a_polyhedron_can_be_built_for_a_site(self):
        from facet.core import cif
        from facet.gl.scene import build_scene

        structure = cif.read(SAMPLE)
        site = structure.site_by_label("Bi2")
        scene = build_scene(structure, polyhedron_site=site)
        assert scene.n_poly_triangles > 0
        assert len(scene.poly_vertices) == len(scene.poly_normals)
        assert len(scene.poly_vertices) % 3 == 0

    def test_space_filling_draws_larger_atoms_than_ball_and_stick(self):
        from facet.core import cif
        from facet.gl.scene import build_scene

        structure = cif.read(SAMPLE)
        ball = build_scene(structure, style=Style.BALL_AND_STICK)
        space = build_scene(structure, style=Style.SPACE_FILLING)
        assert space.atom_radius.mean() > ball.atom_radius.mean() * 1.5


def test_empty_scene_has_a_usable_bounding_sphere():
    s = Scene()
    assert s.radius > 0 and s.n_atoms == 0
    s.restyle(0.1)                                  # must not raise


# ---------------------------------------------------------------------------
# the rotation centre
# ---------------------------------------------------------------------------

def _framed(radius=7.5, centre=(1.0, 2.0, 3.0)):
    from facet.gl.camera import Camera

    camera = Camera()
    camera.frame(np.array(centre, float), radius)
    return camera


def test_the_pivot_and_the_scene_are_separate_points():
    """They coincide until someone chooses a centre, and then they do not.

    Every depth the camera computes used to be measured from the pivot, which
    is only the scene centre while nothing has moved it.
    """
    camera = _framed()
    assert camera.scene_center == pytest.approx(camera.target)
    assert camera.scene_depth() == pytest.approx(camera.distance, abs=1e-12)

    camera.center_on(np.array([9.0, -4.0, 2.0]))
    assert camera.target == pytest.approx([9.0, -4.0, 2.0])
    assert camera.scene_center == pytest.approx([1.0, 2.0, 3.0])


def test_centring_is_a_pure_pan_under_perspective():
    """Choosing a centre must not make anything grow or shrink.

    The eye moves along its own axis by the depth the new centre gains, so
    every drawn point keeps the depth it had. Without that correction the whole
    structure lurches toward or away from the viewer.
    """
    camera = _framed()
    rng = np.random.default_rng(5)
    points = rng.normal(scale=4.0, size=(200, 3)) + np.array([1.0, 2.0, 3.0])

    def depths(cam):
        homogeneous = np.hstack([points, np.ones((len(points), 1))])
        return -(homogeneous @ cam.view_matrix().T)[:, 2]

    before = depths(camera)
    camera.center_on(points[7])
    after = depths(camera)
    assert np.abs(before - after).max() < 1e-9

    # and the chosen point is now in the middle of the view
    projected = camera.project(np.array([points[7]]), 800, 600)[0]
    assert projected[0] == pytest.approx(400.0, abs=1e-6)
    assert projected[1] == pytest.approx(300.0, abs=1e-6)


def test_centring_does_not_rescale_an_orthographic_view():
    """Under an orthographic projection `distance` is the zoom, not a depth.

    `half_height` is `distance * tan(fov/2)` in both projections, and the
    QPainter tier -- which is also the SVG and PDF exporter -- reads it for its
    own scale. Applying the perspective depth correction here would silently
    rescale every orthographic figure.
    """
    camera = _framed()
    camera.orthographic = True
    before = camera.half_height()
    camera.center_on(np.array([9.0, -4.0, 2.0]))
    assert camera.half_height() == pytest.approx(before, rel=1e-12)


def test_the_chosen_point_stays_at_the_centre_under_rotation():
    camera = _framed()
    point = np.array([6.0, 7.0, -1.0])
    camera.center_on(point)
    for _ in range(24):
        camera.drag_rotate(100.0, 100.0, 160.0, 60.0, 900, 700)
        projected = camera.project(np.array([point]), 900, 700)[0]
        assert projected[0] == pytest.approx(450.0, abs=1e-6)
        assert projected[1] == pytest.approx(350.0, abs=1e-6)


def test_an_off_centre_pivot_does_not_clip_the_structure():
    """The window is measured around the scene, not around the pivot.

    Measured on a real structure before the fix: 6 of 68 drawn points fell
    outside near/far with the pivot on the outermost atom, and because the
    QPainter tier does not clip at all, the exported figure would have kept
    drawing what the view had dropped.
    """
    camera = _framed(radius=7.5, centre=(0.0, 0.0, 0.0))
    rng = np.random.default_rng(3)
    shell = rng.normal(size=(400, 3))
    shell /= np.linalg.norm(shell, axis=1)[:, None]
    points = shell * 7.5                       # the extremes of the scene

    camera.center_on(points[0])                # pivot on the surface
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    for _ in range(200):
        camera.drag_rotate(0.0, 0.0, float(rng.uniform(-300, 300)),
                           float(rng.uniform(-300, 300)), 900, 700)
        near, far = camera.clip_planes()
        depth = -(homogeneous @ camera.view_matrix().T)[:, 2]
        assert (depth >= near - 1e-9).all() and (depth <= far + 1e-9).all()


def test_the_fog_window_follows_the_scene_too():
    camera = _framed()
    near, far = camera.fog_range()
    assert near == pytest.approx(camera.distance - camera.scene_radius * 0.55)
    assert far == pytest.approx(camera.distance + camera.scene_radius * 1.25)

    camera.center_on(np.array([9.0, -4.0, 2.0]))
    moved_near, moved_far = camera.fog_range()
    depth = camera.scene_depth()
    assert moved_near == pytest.approx(depth - camera.scene_radius * 0.55)
    assert moved_far == pytest.approx(depth + camera.scene_radius * 1.25)


def test_set_bounds_moves_nothing():
    """It adopts what is drawn; it must not move the camera."""
    camera = _framed()
    before = (camera.target.copy(), camera.distance,
              camera.orientation.copy())
    camera.set_bounds(np.array([40.0, 0.0, 0.0]), 22.0, 1.5)
    assert camera.target == pytest.approx(before[0])
    assert camera.distance == pytest.approx(before[1])
    assert camera.orientation == pytest.approx(before[2])
    assert camera.scene_center == pytest.approx([40.0, 0.0, 0.0])
    assert camera.scene_radius == pytest.approx(22.0)
    assert camera.min_distance == pytest.approx(1.5)


def test_for_eye_carries_every_field():
    """A hand-written field list is where the next field gets forgotten.

    A field left out reverts to its default in one eye only, which shows as the
    two eyes clipping or cueing differently -- the hardest stereo fault to see.
    """
    import dataclasses

    from facet.gl.camera import Camera

    camera = _framed()
    camera.center_on(np.array([5.0, 5.0, 5.0]))
    camera.min_distance = 0.9
    camera.rotation_sense = -1.0
    camera.fov = 31.0

    for eye in (-1, 1):
        other = camera.for_eye(eye)
        for f in dataclasses.fields(Camera):
            if f.name == "orientation":
                continue                       # deliberately different
            mine, theirs = getattr(camera, f.name), getattr(other, f.name)
            if isinstance(mine, np.ndarray):
                assert theirs == pytest.approx(mine), f.name
            else:
                assert theirs == mine, f.name


def test_zoom_cannot_put_the_eye_inside_an_atom():
    camera = _framed()
    camera.min_distance = 2.0
    for _ in range(80):
        camera.zoom(4.0)
    assert camera.distance >= 2.0 - 1e-9
