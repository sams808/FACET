"""Stereo viewing, and the ordering of transparent surfaces.

Stereo is tested on the pixels, because that is all it is: the per-eye rendering
is just a displaced camera, and the part that could be wrong is which channel
came from which eye. The camera side is tested by its own invariants -- both eyes
must still look at the target, and the separation must be an angle rather than a
distance so the depth impression survives zooming.

Transparency ordering is tested by the order the triangles end up in, not by
comparing images: a picture would only show that something changed, while the
order is the thing that decides whether the blend is right.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from conftest import dispose, sample_cif

from facet.gl import stereo as S
from facet.gl.camera import Camera

SAMPLE = sample_cif("1526458", "1526458_Bi2O3.cif")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def structure():
    if not SAMPLE.is_file():
        pytest.skip("the sample structure is not present")
    from facet.core import cif

    return cif.read(SAMPLE)


def _solid(qapp, width, height, rgb, alpha=255):
    from PySide6.QtGui import QImage

    image = QImage(width, height, QImage.Format_RGBA8888)
    image.fill(0)
    array = S._array(image)
    array[..., 0] = rgb[0]
    array[..., 1] = rgb[1]
    array[..., 2] = rgb[2]
    array[..., 3] = alpha
    return S._image(array)


# ===========================================================================
# the camera side
# ===========================================================================

def test_both_eyes_still_look_at_the_target():
    """Toe-in stereo: the two views converge on the framing point.

    If they did not, the pair would have a horizontal offset at the centre of
    the image and would not fuse without eye strain.
    """
    camera = Camera()
    camera.frame(np.array([1.0, -2.0, 0.5]), 6.0)
    target = np.append(camera.target, 1.0)

    for eye in (-1, 0, +1):
        view, projection = camera.eye_matrices(1.6, eye)
        clip = projection @ view @ target
        assert abs(clip[0] / clip[3]) < 1e-9
        assert abs(clip[1] / clip[3]) < 1e-9


def test_the_two_eyes_see_different_views():
    camera = Camera()
    camera.frame(np.zeros(3), 5.0)
    left = camera.eye_matrices(1.5, -1)[0]
    right = camera.eye_matrices(1.5, +1)[0]
    assert not np.allclose(left, right)
    # and each differs from the centre view
    centre = camera.matrices(1.5)[0]
    assert not np.allclose(left, centre)
    assert not np.allclose(right, centre)


def test_eye_zero_is_the_camera_itself():
    camera = Camera()
    camera.frame(np.zeros(3), 5.0)
    assert camera.for_eye(0) is camera
    view, projection = camera.eye_matrices(1.5, 0)
    assert np.allclose(view, camera.view_matrix())


def test_the_two_eyes_sit_symmetrically_about_the_centre():
    """The guarantee toe-in actually makes: the eye positions are symmetric.

    Both eyes are the same distance from the target and displaced equally and
    oppositely across the centre view direction. What is *not* exactly symmetric
    is where an off-axis point lands in each image: rotating about the target
    moves such a point slightly towards one eye and away from the other, and the
    perspective divide then splits the total parallax unevenly between them. That
    is second order -- for a point at the framing radius it is a ten-thousandth
    of the frame -- and it is a property of toe-in, not a fault. The test below
    pins the total parallax, which is what the eyes actually fuse.
    """
    camera = Camera()
    camera.frame(np.zeros(3), 5.0)
    centre = camera.eye()
    left = camera.for_eye(-1).eye()
    right = camera.for_eye(+1).eye()

    # equidistant from the target
    for eye in (left, right):
        assert np.linalg.norm(eye - camera.target) == pytest.approx(
            np.linalg.norm(centre - camera.target), rel=1e-12)
    # and displaced oppositely across the view direction
    forward = (centre - camera.target) / np.linalg.norm(centre - camera.target)
    across_left = (left - centre) - np.dot(left - centre, forward) * forward
    across_right = (right - centre) - np.dot(right - centre, forward) * forward
    assert np.allclose(across_left, -across_right, atol=1e-12)
    assert np.linalg.norm(across_left) > 1e-6


def test_an_off_axis_point_shifts_in_opposite_directions(structure=None):
    """The two eyes must move a point the opposite way, or there is no depth."""
    camera = Camera()
    camera.frame(np.zeros(3), 5.0)
    point = np.array([1.0, 0.0, 0.0, 1.0])

    def ndc_x(eye):
        view, projection = camera.eye_matrices(1.0, eye)
        clip = projection @ view @ point
        return clip[0] / clip[3]

    centre = ndc_x(0)
    left_shift = ndc_x(-1) - centre
    right_shift = ndc_x(+1) - centre
    assert left_shift * right_shift < 0, "both eyes shifted the same way"
    # the uneven split is second order: within a factor of three, not equal
    assert 0.3 < abs(left_shift / right_shift) < 3.0


def test_the_separation_is_an_angle_not_a_distance():
    """The parallax must be the same fraction of the frame at any zoom.

    Expressed as a distance, the stereo effect would collapse on a small cell
    and become unusable on a large one. Expressed as an angle about the target,
    it does not: framing a structure ten times larger gives the same shift on
    screen.
    """
    def parallax(radius):
        camera = Camera()
        camera.frame(np.zeros(3), radius)
        offset = np.array([radius * 0.5, 0.0, 0.0, 1.0])
        shifts = []
        for eye in (-1, +1):
            view, projection = camera.eye_matrices(1.0, eye)
            clip = projection @ view @ offset
            shifts.append(clip[0] / clip[3])
        return shifts[1] - shifts[0]

    small, large = parallax(1.0), parallax(100.0)
    assert small != pytest.approx(0.0, abs=1e-6)
    assert small == pytest.approx(large, rel=1e-6)


def test_a_larger_separation_gives_more_parallax():
    camera = Camera()
    camera.frame(np.zeros(3), 5.0)
    point = np.array([2.0, 0.0, 0.0, 1.0])

    def parallax(separation):
        shifts = []
        for eye in (-1, +1):
            view, projection = camera.eye_matrices(1.0, eye, separation)
            clip = projection @ view @ point
            shifts.append(clip[0] / clip[3])
        return abs(shifts[1] - shifts[0])

    assert parallax(0.5) < parallax(1.2) < parallax(3.0)


def test_a_zero_separation_gives_no_parallax():
    camera = Camera()
    camera.frame(np.zeros(3), 5.0)
    left = camera.eye_matrices(1.0, -1, 0.0)[0]
    right = camera.eye_matrices(1.0, +1, 0.0)[0]
    assert np.allclose(left, right)


def test_for_eye_does_not_disturb_the_original_camera():
    camera = Camera()
    camera.frame(np.array([1.0, 2.0, 3.0]), 4.0)
    before = (camera.target.copy(), camera.distance,
              camera.orientation.copy(), camera.fov)
    camera.for_eye(-1)
    camera.for_eye(+1)
    assert np.array_equal(camera.target, before[0])
    assert camera.distance == before[1]
    assert np.array_equal(camera.orientation, before[2])
    assert camera.fov == before[3]


def test_the_eye_cameras_keep_the_projection_settings():
    camera = Camera()
    camera.frame(np.zeros(3), 5.0)
    camera.orthographic = True
    camera.fov = 8.0
    for eye in (-1, +1):
        shifted = camera.for_eye(eye)
        assert shifted.orthographic
        assert shifted.fov == 8.0
        assert shifted.scene_radius == camera.scene_radius
        assert shifted.distance == camera.distance


# ===========================================================================
# combining the pair
# ===========================================================================

def test_anaglyph_takes_red_from_the_left_and_cyan_from_the_right(qapp):
    left = _solid(qapp, 8, 6, (200, 50, 50))
    right = _solid(qapp, 8, 6, (50, 200, 90))
    out = S._array(S.anaglyph(left, right))
    assert out[0, 0].tolist() == [200, 200, 90, 255]
    # every pixel, not just the first
    assert np.all(out[..., 0] == 200)
    assert np.all(out[..., 1] == 200)
    assert np.all(out[..., 2] == 90)


def test_the_greyscale_anaglyph_gives_both_eyes_the_same_luminance(qapp):
    """Rec. 709 luminance, so the greyscale form keeps the relative brightness.

    Its purpose is to remove the retinal rivalry a saturated colour anaglyph
    causes: a red object is nearly invisible to the eye behind the red filter.
    """
    left = _solid(qapp, 4, 4, (200, 50, 50))
    right = _solid(qapp, 4, 4, (50, 200, 90))
    out = S._array(S.anaglyph(left, right, greyscale=True))

    left_luma = 0.2126 * 200 + 0.7152 * 50 + 0.0722 * 50
    right_luma = 0.2126 * 50 + 0.7152 * 200 + 0.0722 * 90
    assert out[0, 0, 0] == int(left_luma)
    assert out[0, 0, 1] == int(right_luma)
    assert out[0, 0, 2] == int(right_luma)
    # cyan channels are equal, which is what makes it grey to that eye
    assert out[0, 0, 1] == out[0, 0, 2]


def test_anaglyph_keeps_alpha_where_either_eye_drew(qapp):
    left = _solid(qapp, 4, 4, (10, 10, 10), alpha=0)
    right = _solid(qapp, 4, 4, (10, 10, 10), alpha=255)
    out = S._array(S.anaglyph(left, right))
    assert np.all(out[..., 3] == 255)


def test_side_by_side_places_the_eyes_in_order(qapp):
    left = _solid(qapp, 8, 6, (200, 50, 50))
    right = _solid(qapp, 8, 6, (50, 200, 90))
    out = S.side_by_side(left, right, gap=4)
    assert (out.width(), out.height()) == S.output_size(
        S.Mode.SIDE_BY_SIDE, 8, 6, 4)
    array = S._array(out)
    assert array[0, 0, :3].tolist() == [200, 50, 50]
    assert array[0, -1, :3].tolist() == [50, 200, 90]


def test_the_crossed_pair_swaps_them(qapp):
    left = _solid(qapp, 8, 6, (200, 50, 50))
    right = _solid(qapp, 8, 6, (50, 200, 90))
    array = S._array(S.side_by_side(left, right, crossed=True, gap=4))
    assert array[0, 0, :3].tolist() == [50, 200, 90]
    assert array[0, -1, :3].tolist() == [200, 50, 50]


def test_the_gap_is_drawn_and_is_the_width_asked_for(qapp):
    left = _solid(qapp, 6, 4, (255, 255, 255))
    right = _solid(qapp, 6, 4, (255, 255, 255))
    array = S._array(S.side_by_side(left, right, gap=5,
                                    gap_color=(0, 0, 0)))
    assert array.shape[1] == 6 + 5 + 6
    middle = array[0, 6:11, :3]
    assert np.all(middle == 0)


def test_a_zero_gap_is_allowed(qapp):
    left = _solid(qapp, 6, 4, (255, 0, 0))
    right = _solid(qapp, 6, 4, (0, 255, 0))
    out = S.side_by_side(left, right, gap=0)
    assert out.width() == 12


def test_mismatched_eyes_are_refused(qapp):
    left = _solid(qapp, 8, 6, (1, 1, 1))
    right = _solid(qapp, 8, 7, (1, 1, 1))
    with pytest.raises(ValueError, match="same size"):
        S.anaglyph(left, right)
    with pytest.raises(ValueError, match="same size"):
        S.side_by_side(left, right)


def test_combine_dispatches_every_mode(qapp):
    left = _solid(qapp, 8, 6, (200, 50, 50))
    right = _solid(qapp, 8, 6, (50, 200, 90))
    for mode in S.Mode:
        out = S.combine(left, right, mode)
        assert not out.isNull()
        expected = S.output_size(mode, 8, 6)
        if mode is not S.Mode.OFF:
            assert (out.width(), out.height()) == expected


def test_mode_off_returns_the_left_eye_unchanged(qapp):
    left = _solid(qapp, 8, 6, (200, 50, 50))
    right = _solid(qapp, 8, 6, (50, 200, 90))
    out = S.combine(left, right, S.Mode.OFF)
    assert S._array(out)[0, 0, :3].tolist() == [200, 50, 50]


def test_the_returned_image_owns_its_pixels(qapp):
    """QImage does not copy the buffer it is handed.

    Returning one that points at a numpy array about to be collected gives an
    image whose pixels are freed memory -- noise or a crash, not an error. The
    combination must hand back an image that owns its own data.
    """
    left = _solid(qapp, 16, 16, (200, 50, 50))
    right = _solid(qapp, 16, 16, (50, 200, 90))
    out = S.anaglyph(left, right)
    import gc

    del left, right
    gc.collect()
    array = S._array(out)
    assert array[0, 0, :3].tolist() == [200, 200, 90]


def test_mode_properties_are_consistent():
    assert not S.Mode.OFF.needs_two_eyes
    for mode in S.Mode:
        if mode is not S.Mode.OFF:
            assert mode.needs_two_eyes
    assert S.Mode.ANAGLYPH.is_anaglyph
    assert S.Mode.ANAGLYPH_GREY.is_anaglyph
    assert not S.Mode.SIDE_BY_SIDE.is_anaglyph
    assert not S.Mode.OFF.is_anaglyph


# ===========================================================================
# stereo through the view
# ===========================================================================

def test_the_view_renders_a_stereo_pair_on_whatever_tier(qapp, structure):
    """The software tier must do stereo too: no graphics card required."""
    from facet.gl.scene import build_scene
    from facet.gl.view import StructureView

    widget = StructureView()
    widget.resize(160, 120)
    widget.set_scene(build_scene(structure, polyhedron_site=0))
    widget.set_stereo(S.Mode.ANAGLYPH)

    left, right = widget.stereo_pair(160, 120)
    assert not left.isNull() and not right.isNull()
    assert (left.width(), left.height()) == (160, 120)
    combined = S.combine(left, right, S.Mode.ANAGLYPH)
    assert not combined.isNull()
    dispose(widget)


def test_grab_image_honours_the_stereo_mode(qapp, structure):
    from facet.gl.scene import build_scene
    from facet.gl.view import StructureView

    widget = StructureView()
    widget.resize(120, 90)
    widget.set_scene(build_scene(structure))

    widget.set_stereo(S.Mode.SIDE_BY_SIDE)
    wide = widget.grab_image(120, 90, supersample=1)
    assert wide.width() > wide.height()
    assert wide.width() >= 240

    widget.set_stereo(S.Mode.OFF)
    plain = widget.grab_image(120, 90, supersample=1)
    assert plain.width() <= 130
    dispose(widget)


def test_stereo_off_is_the_default_and_costs_one_render(qapp):
    from facet.gl.view import StructureView

    widget = StructureView()
    assert widget.stereo is S.Mode.OFF
    assert not widget.stereo.needs_two_eyes
    dispose(widget)


# ===========================================================================
# transparency ordering
# ===========================================================================

def test_transparent_triangles_are_ordered_back_to_front(structure):
    """The property that makes alpha blending correct.

    Blending is not commutative, so a transparent surface drawn in upload order
    is drawn wrong: of two overlapping polyhedra, whichever sits later in the
    buffer appears in front, for no reason the user can see. After sorting, every
    triangle must be at least as far away as the one drawn after it.
    """
    from facet.gl.scene import build_scene

    scene = build_scene(structure,
                        polyhedron_sites=structure.cation_sites)
    if not scene.n_poly_triangles:
        pytest.skip("no polyhedra were built for this structure")

    # the sort is a pure function of the view direction; exercise it directly
    from facet.gl import renderer as renderer_mod

    class Bare:
        """Just the sorting behaviour, with the GL upload stubbed out."""
        sort_transparency = True
        _sort_key = None
        _make_vao = staticmethod(lambda *a, **k: None)
        _sort_transparent = renderer_mod.Renderer._sort_transparent

    for direction in (np.array([[0, 0, 0, 0], [0, 0, 0, 0],
                                [0.0, 0.0, 1.0, 0.0], [0, 0, 0, 1]]),
                      np.array([[0, 0, 0, 0], [0, 0, 0, 0],
                                [0.577, 0.577, 0.577, 0.0], [0, 0, 0, 1]]),
                      np.array([[0, 0, 0, 0], [0, 0, 0, 0],
                                [-0.3, 0.8, -0.5, 0.0], [0, 0, 0, 1]])):
        bare = Bare()
        bare._sort_key = None
        bare._sort_transparent(direction, scene)

        forward = direction[2, :3] / np.linalg.norm(direction[2, :3])
        depth = scene.poly_vertices.reshape(-1, 3, 3).mean(axis=1) @ forward
        assert np.all(np.diff(depth) >= -1e-5), \
            "polyhedron triangles are not back to front"


def test_the_sort_is_cached_against_the_view_direction(structure):
    """Dragging the cutoff must not re-upload the transparent geometry."""
    from facet.gl import renderer as renderer_mod
    from facet.gl.scene import build_scene

    scene = build_scene(structure,
                        polyhedron_sites=structure.cation_sites)
    if not scene.n_poly_triangles:
        pytest.skip("no polyhedra were built")

    uploads = []

    class Bare:
        sort_transparency = True
        _sort_key = None

        def _make_vao(self, key, arrays, count):
            uploads.append(key)

        _sort_transparent = renderer_mod.Renderer._sort_transparent

    view = np.eye(4)
    bare = Bare()
    bare._sort_transparent(view, scene)
    first = len(uploads)
    assert first > 0

    bare._sort_transparent(view, scene)          # same view
    assert len(uploads) == first, "the same view re-sorted"

    turned = np.eye(4)
    turned[2, :3] = [0.5, 0.5, 0.707]
    bare._sort_transparent(turned, scene)
    assert len(uploads) > first, "a new view did not re-sort"


def test_sorting_can_be_switched_off(structure):
    from facet.gl import renderer as renderer_mod
    from facet.gl.scene import build_scene

    scene = build_scene(structure,
                        polyhedron_sites=structure.cation_sites)
    if not scene.n_poly_triangles:
        pytest.skip("no polyhedra were built")
    before = scene.poly_vertices.copy()

    class Bare:
        sort_transparency = False
        _sort_key = None
        _make_vao = staticmethod(lambda *a, **k: None)
        _sort_transparent = renderer_mod.Renderer._sort_transparent

    Bare()._sort_transparent(np.eye(4), scene)
    assert np.array_equal(scene.poly_vertices, before)


def test_sorting_preserves_every_triangle(structure):
    """A sort must reorder, never lose or duplicate."""
    from facet.gl import renderer as renderer_mod
    from facet.gl.scene import build_scene

    scene = build_scene(structure,
                        polyhedron_sites=structure.cation_sites)
    if not scene.n_poly_triangles:
        pytest.skip("no polyhedra were built")

    before = scene.poly_vertices.reshape(-1, 3, 3).copy()

    class Bare:
        sort_transparency = True
        _sort_key = None
        _make_vao = staticmethod(lambda *a, **k: None)
        _sort_transparent = renderer_mod.Renderer._sort_transparent

    view = np.eye(4)
    view[2, :3] = [0.3, -0.6, 0.74]
    Bare()._sort_transparent(view, scene)
    after = scene.poly_vertices.reshape(-1, 3, 3)

    assert len(after) == len(before)
    assert len(scene.poly_normals) == len(scene.poly_vertices)

    def signature(triangles):
        return sorted(tuple(np.round(t.reshape(-1), 4)) for t in triangles)

    assert signature(after) == signature(before)


def test_the_normals_travel_with_their_triangles(structure):
    """Sorting positions without their normals would light the surface wrongly."""
    from facet.gl import renderer as renderer_mod
    from facet.gl.scene import build_scene

    scene = build_scene(structure,
                        polyhedron_sites=structure.cation_sites)
    if not scene.n_poly_triangles:
        pytest.skip("no polyhedra were built")

    pairs_before = {
        tuple(np.round(v.reshape(-1), 4)): tuple(np.round(n.reshape(-1), 4))
        for v, n in zip(scene.poly_vertices.reshape(-1, 3, 3),
                        scene.poly_normals.reshape(-1, 3, 3))}

    class Bare:
        sort_transparency = True
        _sort_key = None
        _make_vao = staticmethod(lambda *a, **k: None)
        _sort_transparent = renderer_mod.Renderer._sort_transparent

    view = np.eye(4)
    view[2, :3] = [0.1, 0.2, 0.97]
    Bare()._sort_transparent(view, scene)

    for v, n in zip(scene.poly_vertices.reshape(-1, 3, 3),
                    scene.poly_normals.reshape(-1, 3, 3)):
        key = tuple(np.round(v.reshape(-1), 4))
        assert pairs_before[key] == tuple(np.round(n.reshape(-1), 4))


def test_planes_are_ordered_as_whole_sheets(structure):
    """A plane is flat, so its own triangles cannot occlude each other."""
    from facet.core import planes as planes_mod
    from facet.gl import renderer as renderer_mod
    from facet.gl.scene import build_scene

    scene = build_scene(structure, lattice_planes=[
        planes_mod.LatticePlane(0, 0, 1, offset=0.2, color=(1.0, 0, 0)),
        planes_mod.LatticePlane(0, 0, 1, offset=0.8, color=(0, 1.0, 0)),
        planes_mod.LatticePlane(1, 0, 0, offset=0.5, color=(0, 0, 1.0))])
    assert scene.n_planes == 3

    class Bare:
        sort_transparency = True
        _sort_key = None
        _make_vao = staticmethod(lambda *a, **k: None)
        _sort_transparent = renderer_mod.Renderer._sort_transparent

    view = np.eye(4)
    view[2, :3] = [0.0, 0.0, 1.0]
    Bare()._sort_transparent(view, scene)

    forward = view[2, :3]
    depths = [float(np.mean(np.asarray(mesh[0], float) @ forward))
              for mesh in scene.plane_meshes]
    assert depths == sorted(depths)
    # all three are still there, with their colours
    assert {mesh[2] for mesh in scene.plane_meshes} == {
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)}


def test_the_software_tier_draws_polyhedra(qapp, structure):
    """A polyhedron is the point of the program; the fallback must show them."""
    from PySide6.QtGui import QImage, QPainter

    from facet.gl.camera import Camera
    from facet.gl.painter import PainterRenderer
    from facet.gl.scene import build_scene

    def colours(scene):
        image = QImage(320, 240, QImage.Format_ARGB32)
        image.fill(0)
        painter = QPainter(image)
        camera = Camera()
        camera.frame(scene.center, scene.radius)
        try:
            PainterRenderer().render(painter, scene, camera, 320, 240)
        finally:
            painter.end()
        seen = set()
        for y in range(0, 240, 4):
            for x in range(0, 320, 4):
                pixel = image.pixelColor(x, y)
                seen.add((pixel.red(), pixel.green(), pixel.blue()))
        return seen

    site = structure.site_by_label("Bi1")
    if site is None:
        site = 0
    plain = colours(build_scene(structure))
    with_polyhedron = colours(build_scene(structure, polyhedron_site=site))
    assert with_polyhedron != plain, "the polyhedron was not drawn"
    assert len(with_polyhedron) > len(plain)


# ---------------------------------------------------------------------------
# where the two pictures are
# ---------------------------------------------------------------------------
#
# The composition was always right; what was missing was any statement of where
# it put the two images, so everything that had to know -- the blit, the labels
# drawn over it, the click that picks an atom out of it -- guessed, and guessed
# differently.

SIDE_MODES = (S.Mode.SIDE_BY_SIDE, S.Mode.CROSS_EYED)
FLAT_MODES = (S.Mode.OFF, S.Mode.ANAGLYPH, S.Mode.ANAGLYPH_GREY)


@pytest.mark.parametrize("mode", list(S.Mode))
@pytest.mark.parametrize("area", [(738, 552), (1201, 800), (321, 321),
                                  (1920, 1080), (64, 48)])
def test_a_pair_composed_for_an_area_fits_in_it(mode, area):
    """It must fit, or the blit scales it and every pixel measurement is out."""
    width, height = area
    ew, eh, gap = S.pane_size(mode, width, height)
    cw, ch = S.output_size(mode, ew, eh, gap)
    assert cw <= width and ch <= height
    assert ew >= 1 and eh >= 1


@pytest.mark.parametrize("mode", SIDE_MODES)
@pytest.mark.parametrize("area", [(738, 552), (1201, 800), (1920, 1080)])
def test_each_eye_keeps_the_proportions_of_the_area(mode, area):
    """Two copies of the same picture, not two tall slices of it.

    Giving each eye half the width and the full height crops both of them, so a
    wide structure is cut off in both eyes at once -- and the aspect of what is
    drawn no longer matches the window it was framed in.
    """
    width, height = area
    ew, eh, gap = S.pane_size(mode, width, height)
    assert ew / eh == pytest.approx(width / height, rel=0.01)
    assert S.output_size(mode, ew, eh, gap)[0] == pytest.approx(width, abs=1)


@pytest.mark.parametrize("mode", FLAT_MODES)
def test_a_superimposed_mode_is_one_pane_and_the_fused_camera(mode):
    """An anaglyph has one picture, and the viewer points at the fused image.

    That position belongs to the undisplaced camera, not to either eye, which is
    why the pane reports eye 0 rather than picking one of them.
    """
    panes = S.panes(mode, 800, 600)
    assert len(panes) == 1
    eye, x, y, w, h = panes[0]
    assert (eye, x, y, w, h) == (0, 0, 0, 800, 600)


@pytest.mark.parametrize("mode", SIDE_MODES)
def test_the_crossed_layout_swaps_which_eye_is_on_the_left(mode):
    panes = S.panes(mode, 400, 300, 8)
    assert [p[0] for p in panes] == (
        [+1, -1] if mode is S.Mode.CROSS_EYED else [-1, +1])
    assert [p[1] for p in panes] == [0, 408]


@pytest.mark.parametrize("mode", list(S.Mode))
def test_every_point_of_a_pane_locates_back_to_it(mode):
    """The round trip the picking depends on."""
    ew, eh, gap = S.pane_size(mode, 738, 552)
    for eye, x0, y0, w, h in S.panes(mode, ew, eh, gap):
        for fx, fy in ((0.0, 0.0), (0.5, 0.5), (0.999, 0.999), (0.1, 0.9)):
            x, y = x0 + fx * (w - 1), y0 + fy * (h - 1)
            found = S.locate(mode, ew, eh, x, y, gap)
            assert found is not None, (mode, x, y)
            assert found[0] == eye
            assert found[1] == pytest.approx(x - x0)
            assert found[2] == pytest.approx(y - y0)


@pytest.mark.parametrize("mode", SIDE_MODES)
def test_the_seam_belongs_to_neither_eye(mode):
    """There is nothing drawn there, so there is nothing to pick."""
    ew, eh, gap = S.pane_size(mode, 738, 552)
    assert gap >= 1
    for x in range(ew, ew + gap):
        assert S.locate(mode, ew, eh, x, eh // 2, gap) is None
    assert S.locate(mode, ew, eh, -1, 5, gap) is None
    assert S.locate(mode, ew, eh, 2 * ew + gap + 1, 5, gap) is None


def test_the_gap_can_be_given_the_colour_of_the_ground(qapp):
    """A black bar down the middle of a white figure reads as part of it."""
    left = _solid(qapp, 40, 30, (200, 40, 40))
    right = _solid(qapp, 40, 30, (40, 40, 200))
    joined = S.side_by_side(left, right, gap=6, gap_color=(255, 255, 255))
    assert joined.width() == 86
    middle = joined.pixelColor(43, 15)
    assert (middle.red(), middle.green(), middle.blue()) == (255, 255, 255)
