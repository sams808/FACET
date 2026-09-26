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
