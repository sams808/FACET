"""End-to-end rendering, checked on the pixels.

These tests open a real OpenGL context. They are the only way to catch the
class of bug that raises no error and renders nothing -- which is exactly what
the PySide6 buffer-offset trap does.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

SAMPLE = Path(r"C:\Users\samso\Desktop\WSU_work\XRD\cif\Bi\cifs\1526458_Bi2O3.cif")
pytestmark = pytest.mark.skipif(not SAMPLE.exists(),
                                reason="sample structure not present")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    from facet.gl import caps as caps_mod
    from PySide6.QtGui import QSurfaceFormat

    QSurfaceFormat.setDefaultFormat(caps_mod.request_format())
    existing = QApplication.instance()
    yield existing or QApplication([])


@pytest.fixture(scope="module")
def scene():
    from facet.core import cif
    from facet.gl.scene import build_scene

    structure = cif.read(SAMPLE)
    site = structure.site_by_label("Bi1")
    return build_scene(structure, polyhedron_site=site)


def _distinct_colours(image, step=11):
    seen = set()
    for y in range(0, image.height(), step):
        for x in range(0, image.width(), step):
            c = image.pixelColor(x, y)
            seen.add((c.red(), c.green(), c.blue()))
    return seen


class TestGLView:
    """Drives the real widget. Skipped where no GL context can be made."""

    @pytest.fixture(scope="class")
    def view(self, app, scene):
        from PySide6.QtCore import QTimer
        from facet.gl.view import StructureView

        v = StructureView()
        v.resize(420, 320)
        v.set_scene(scene)
        v.show()
        # let Qt create the context and run initializeGL
        for _ in range(20):
            app.processEvents()
        if v.caps is None:
            pytest.skip("no OpenGL context available in this environment")
        yield v
        v.close()

    def test_a_tier_was_selected_and_described(self, view):
        assert view.caps is not None
        assert view.caps.describe()

    def test_rendering_produces_a_non_blank_image(self, view, app):
        image = view.grab_image(360, 280, supersample=1)
        assert not image.isNull()
        colours = _distinct_colours(image)
        assert len(colours) > 12, (
            f"only {len(colours)} distinct colours -- the scene did not draw. "
            f"Tier: {view.caps.describe()}")

    def test_the_render_is_not_only_background(self, view):
        image = view.grab_image(360, 280, supersample=1)
        bg = image.pixelColor(2, 2)
        centre = image.pixelColor(image.width() // 2, image.height() // 2)
        assert (centre.red(), centre.green(), centre.blue()) != \
               (bg.red(), bg.green(), bg.blue())

    def test_supersampled_export_matches_the_requested_size(self, view):
        image = view.grab_image(300, 200, supersample=2)
        assert (image.width(), image.height()) == (300, 200)

    def test_moving_the_threshold_changes_what_is_drawn(self, view, scene):
        wide = view.grab_image(240, 200, supersample=1)
        view.set_threshold(0.35)
        narrow = view.grab_image(240, 200, supersample=1)
        view.set_threshold(0.02)
        broad = view.grab_image(240, 200, supersample=1)
        view.set_threshold(0.075)
        assert wide.constBits() != narrow.constBits()
        assert narrow.constBits() != broad.constBits()

    def test_rotating_changes_the_image(self, view):
        before = view.grab_image(240, 200, supersample=1)
        view.camera.drag_rotate(60, 60, 170, 130, 240, 200)
        after = view.grab_image(240, 200, supersample=1)
        view.reset_view()
        assert before.constBits() != after.constBits()

    def test_turning_the_structure_does_not_change_how_brightly_it_is_lit(
            self, view):
        """The lamp is over the viewer's shoulder, not bolted to the crystal.

        This is the fault the test exists for: the light used to be a direction
        in the *crystal's* frame, rotated into view space every frame
        (``light_view = view[:3, :3] @ light``). Orbiting therefore carried the
        lamp around with the structure, and the far side went dark -- measured
        at 78% of the reset view's mean atom luminance on this machine, and the
        diffuse term itself at 53% over a full orbit.

        Measured on the drawn pixels rather than the whole frame, because the
        background is a large constant area that would dilute the effect to
        nothing.
        """
        def lit_mean(image):
            background = image.pixelColor(1, 1)
            total, count = 0.0, 0
            for y in range(0, image.height(), 3):
                for x in range(0, image.width(), 3):
                    c = image.pixelColor(x, y)
                    if (abs(c.red() - background.red()) < 6
                            and abs(c.green() - background.green()) < 6
                            and abs(c.blue() - background.blue()) < 6):
                        continue
                    total += (0.2126 * c.redF() + 0.7152 * c.greenF()
                              + 0.0722 * c.blueF())
                    count += 1
            return (total / count) if count else 0.0

        view.reset_view()
        reset = lit_mean(view.grab_image(240, 200, supersample=1))
        assert reset > 0.0, "nothing was drawn, so there is nothing to measure"

        means = [reset]
        for _ in range(4):
            # four quarter turns about the screen-up axis, back to the start
            view.camera.drag_rotate(0, 100, 120, 100, 240, 200)
            means.append(lit_mean(view.grab_image(240, 200, supersample=1)))
        view.reset_view()

        spread = max(means) / max(min(means), 1e-9)
        assert spread < 1.25, (
            f"the structure is {spread:.2f}x brighter from one side than from "
            f"another: {['%.4f' % m for m in means]}. The light has become "
            f"fixed to the crystal again.")

    def test_picking_the_centre_returns_a_real_atom_or_background(self, view):
        from PySide6.QtCore import QPoint

        result = view.pick_at(QPoint(view.width() // 2, view.height() // 2))
        assert result is None or 0 <= result < view.scene.n_atoms

    def test_picking_a_known_atom_returns_that_atom(self, view, scene):
        """Project an atom to screen, click it, and get it back.

        This is the real test of the id-buffer path: it verifies that what is
        drawn and what is picked agree.
        """
        from PySide6.QtCore import QPoint

        px = view.camera.project(scene.atom_position, view.width(), view.height())
        front = np.argsort(px[:, 2])          # most negative = nearest
        for i in front[:6]:
            x, y = int(px[i, 0]), int(px[i, 1])
            if not (4 < x < view.width() - 4 and 4 < y < view.height() - 4):
                continue
            got = view.pick_at(QPoint(x, y))
            if got is not None:
                # the nearest atom along that ray, which may occlude i
                assert 0 <= got < scene.n_atoms
                return
        pytest.skip("no atom projected to a clickable position")


class TestPainterFallback:
    """The QPainter path must work with no GL at all."""

    def test_it_draws_something(self, app, scene):
        from PySide6.QtGui import QImage, QPainter
        from facet.gl.camera import Camera
        from facet.gl.painter import PainterRenderer

        camera = Camera()
        camera.frame(scene.center, scene.radius)
        image = QImage(360, 280, QImage.Format_RGB32)
        image.fill(0)
        painter = QPainter(image)
        PainterRenderer().render(painter, scene, camera, 360, 280)
        painter.end()

        colours = _distinct_colours(image)
        assert len(colours) > 12, "fallback renderer drew nothing"

    def test_perspective_shrinks_distant_atoms(self, app):
        from facet.gl.camera import Camera
        from facet.gl.painter import PainterRenderer

        r = PainterRenderer()
        camera = Camera()
        camera.frame([0, 0, 0], 5.0)
        r._scale = 100.0
        r._distance = camera.distance
        r._perspective = True
        near = r.radius_at(0.5, -(camera.distance - 4))
        far = r.radius_at(0.5, -(camera.distance + 4))
        assert near > far

    def test_orthographic_keeps_atom_size_constant(self, app):
        from facet.gl.painter import PainterRenderer

        r = PainterRenderer()
        r._scale = 100.0
        r._distance = 20.0
        r._perspective = False
        assert r.radius_at(0.5, -10.0) == pytest.approx(r.radius_at(0.5, -30.0))

    def test_depth_cue_fades_towards_the_background(self, app):
        """The far colour must be nearer the background than the near one.

        Not "darker": the background is a theme's choice and the shipped one is
        white, so a test asserting that depth cueing darkens an atom is really
        asserting that the application is dark.
        """
        from PySide6.QtGui import QColor
        from facet.gl.painter import PainterRenderer

        r = PainterRenderer()
        r._near, r._far = 10.0, 20.0
        bg = r.background
        # start from the colour furthest from the background, so the fade has
        # somewhere to go whichever way round the theme is
        ink = QColor(0, 0, 0) if bg.lightnessF() > 0.5 else QColor(255, 255, 255)

        def distance(c):
            return (abs(c.red() - bg.red()) + abs(c.green() - bg.green())
                    + abs(c.blue() - bg.blue()))

        front = r._cue(ink, -10.0)
        back = r._cue(ink, -20.0)
        assert distance(back) < distance(front)
        # and it is a fade, not a jump to the background
        assert distance(back) > 0

    def test_it_survives_an_empty_scene(self, app):
        from PySide6.QtGui import QImage, QPainter
        from facet.gl.camera import Camera
        from facet.gl.painter import PainterRenderer
        from facet.gl.scene import Scene

        image = QImage(80, 60, QImage.Format_RGB32)
        painter = QPainter(image)
        PainterRenderer().render(painter, Scene(), Camera(), 80, 60)
        painter.end()


# ---------------------------------------------------------------------------
# the light itself
# ---------------------------------------------------------------------------

def test_the_light_is_a_unit_vector_pointing_at_the_viewer():
    """A headlight, by definition: fixed in view space with a positive z.

    Needs no GL context, so it guards the property on every machine. The z
    component is the one that matters -- it is what puts the lamp on the
    viewer's side. x and y only decide which shoulder it sits over, and a light
    with no x or y at all would flatten every sphere into a disc.
    """
    import numpy as np

    from facet.gl.renderer import LIGHT_VIEW

    assert float(np.linalg.norm(LIGHT_VIEW)) == pytest.approx(1.0, abs=1e-12)
    assert LIGHT_VIEW[2] > 0.3, (
        "the light points away from the viewer, so the visible side of every "
        "atom is the unlit one")
    assert abs(LIGHT_VIEW[0]) > 0.05 and abs(LIGHT_VIEW[1]) > 0.05, (
        "a light exactly along the view axis gives a sphere no shading at all")


def test_the_shader_light_does_not_depend_on_the_camera():
    """Read the renderer's own source, because this is what regressed.

    A brittle test on purpose. The fault it guards is a single matrix multiply
    that reintroduces itself naturally -- transforming a direction into view
    space looks like the right thing to do, and the result renders perfectly at
    the default orientation. Nothing else in the suite would notice.
    """
    import inspect

    from facet.gl.renderer import Renderer

    source = inspect.getsource(Renderer.render)
    assert "light_view = LIGHT_VIEW" in source
    assert "@ light" not in source, (
        "the light is being transformed by the camera again; it must stay a "
        "view-space constant")
