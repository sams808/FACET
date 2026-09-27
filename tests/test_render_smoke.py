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
from PySide6.QtCore import QPoint

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


def _row_colours(image, y, step=3):
    return {(c.red(), c.green(), c.blue()) for c in
            (image.pixelColor(x, y) for x in range(0, image.width(), step))}


def _rgb_array(image):
    from PySide6.QtGui import QImage

    converted = image.convertToFormat(QImage.Format_RGBA8888)
    raw = np.frombuffer(memoryview(converted.constBits()), np.uint8,
                        count=converted.height() * converted.width() * 4)
    # copied, not viewed: the array outlives `converted`, and a view into a
    # freed QImage buffer reads as noise or takes the process down
    return raw.reshape(converted.height(), converted.width(), 4)[..., :3].copy()


def _difference(a, b, step=3):
    """How many sampled pixels differ between two images of the same size."""
    if (a.width(), a.height()) != (b.width(), b.height()):
        return -1
    count = 0
    for y in range(0, a.height(), step):
        for x in range(0, a.width(), step):
            if a.pixelColor(x, y) != b.pixelColor(x, y):
                count += 1
    return count


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

    def test_every_stereo_mode_renders(self, view, app):
        """Choosing anaglyph used to crash the application outright.

        Not an exception -- an access violation, exit 139, every time, on the
        real OpenGL tier. `paintGL` renders the two eyes through
        `_render_to_image`, which called `makeCurrent()` and then
        `doneCurrent()`; inside `paintGL` the context is already current, so
        that released it out from under the rest of the method, which then
        opened a QOpenGLPaintDevice on a framebuffer bound to nothing.

        1187 tests passed with this in the shipped build, because none of them
        drove the widget through a stereo mode. This one does, and it is why it
        is here rather than beside the stereo arithmetic in
        tests/test_stereo_transparency.py -- the fault was never in the stereo
        maths.
        """
        from facet.gl import stereo as stereo_mod

        try:
            for mode in stereo_mod.Mode:
                view.set_stereo(mode, 1.5)
                for _ in range(3):
                    app.processEvents()
                image = view.grab_image(200, 160, supersample=1)
                assert not image.isNull(), mode.name
                assert len(_distinct_colours(image)) > 2, mode.name
        finally:
            view.set_stereo(stereo_mod.Mode.OFF, 1.5)
            for _ in range(2):
                app.processEvents()

    def test_a_click_picks_what_is_drawn_under_it(self, view, app):
        """Under a side-by-side mode the widget holds two pictures.

        Each is drawn by its own camera into its own half, and a click belongs
        to whichever half it landed in. The shipped build answered every click
        with the mono camera at the full widget size -- more than a quarter of
        the width out, so clicking an atom picked a different one, or nothing.

        An atom in front of the one aimed at is a correct answer; a click that
        lands on nothing, or on something further away, is not.
        """
        from facet.gl import stereo as stereo_mod

        scene = view.scene
        camera_of = lambda eye: (view.camera if eye == 0 else
                                 view.camera.for_eye(eye, view.stereo_separation))
        # the atoms furthest from the middle, where a missing pane offset shows
        order = np.argsort(-np.linalg.norm(
            scene.atom_position - scene.center, axis=1))[:24]

        try:
            for mode in stereo_mod.Mode:
                view.set_stereo(mode, 1.5)
                for _ in range(3):
                    app.processEvents()
                ox, oy, panes, ew, eh, gap = view._stereo_layout(
                    view.width(), view.height())
                for eye, x0, y0, pw, ph in panes:
                    camera = camera_of(eye)
                    for index in (int(i) for i in order):
                        p = camera.project(
                            np.array([scene.atom_position[index]]), pw, ph)[0]
                        if p[2] >= 0 or not (0 <= p[0] < pw and 0 <= p[1] < ph):
                            continue
                        hit = view.pick_at(QPoint(int(round(ox + x0 + p[0])),
                                                  int(round(oy + y0 + p[1]))))
                        assert hit is not None, (
                            f"{mode.name}, eye {eye}: clicking where "
                            f"{scene.atom_label[index]} is drawn picked nothing")
                        depth = -camera.project(
                            np.array([scene.atom_position[hit]]), pw, ph)[0][2]
                        assert depth <= -p[2] + 1e-6, (
                            f"{mode.name}, eye {eye}: clicking "
                            f"{scene.atom_label[index]} picked "
                            f"{scene.atom_label[hit]}, which is behind it")
        finally:
            view.set_stereo(stereo_mod.Mode.OFF, 1.5)
            for _ in range(2):
                app.processEvents()

    def test_a_bond_across_an_atom_does_not_shield_it(self, view):
        """Clicking the middle of a big atom used to select nothing.

        Bonds were drawn into the pick pass with an id of zero, meaning
        "background" -- but they still wrote depth, so a bond crossing an atom's
        centre did not read as background, it read as a hole. Measured on Bi2O3:
        a Bi drawn 19 px across had a dead cross through the middle of it.
        """
        scene = view.scene
        projected = view.camera.project(scene.atom_position,
                                        view.width(), view.height())
        # the biggest atoms on screen: the ones whose own bonds cross them
        radii = np.array([view._screen_radius(scene.atom_position[i],
                                              scene.atom_radius[i])
                          for i in range(scene.n_atoms)])
        front = projected[:, 2] < 0
        candidates = [i for i in np.argsort(-radii)
                      if front[i] and radii[i] > 6
                      and 0 <= projected[i][0] < view.width()
                      and 0 <= projected[i][1] < view.height()][:12]
        assert candidates, "no atom is drawn large enough to test"

        blind = []
        for i in candidates:
            p = projected[i]
            if view.pick_at(QPoint(int(round(p[0])), int(round(p[1])))) is None:
                blind.append(scene.atom_label[i])
        assert not blind, (
            f"the centre of {len(blind)} visible atom(s) picked nothing: {blind}")

    def test_the_pair_stays_inside_the_widget(self, view, app):
        """Rendered to fit, not rendered full size and squeezed in afterwards.

        The shipped build rendered each eye at the whole widget size and fitted
        the double-width result in, which left the pair at half size in a band
        across the top -- and nothing cleared the rest, so the previous frame
        showed underneath it.
        """
        from facet.gl import stereo as stereo_mod

        try:
            for mode in stereo_mod.Mode:
                view.set_stereo(mode, 1.5)
                ratio = view.devicePixelRatioF()
                w = max(1, int(view.width() * ratio))
                h = max(1, int(view.height() * ratio))
                ox, oy, panes, ew, eh, gap = view._stereo_layout(w, h)
                cw, ch = stereo_mod.output_size(mode, ew, eh, gap)
                assert ox >= 0 and oy >= 0, f"{mode.name}: offset {ox}, {oy}"
                assert ox + cw <= w + 1 and oy + ch <= h + 1, (
                    f"{mode.name}: a {cw}x{ch} pair placed at ({ox}, {oy}) "
                    f"does not fit in a {w}x{h} widget")
                for eye, x0, y0, pw, ph in panes:
                    assert x0 + pw <= cw and y0 + ph <= ch, (
                        f"{mode.name}: eye {eye} runs outside the pair")
        finally:
            view.set_stereo(stereo_mod.Mode.OFF, 1.5)
            for _ in range(2):
                app.processEvents()

    def test_the_margin_around_a_pair_is_the_background(self, view, app):
        """Whatever was on screen before must not survive under the pair."""
        from facet.core import theme as theme_mod
        from facet.gl import stereo as stereo_mod

        view.theme = theme_mod.Theme()
        try:
            view.set_stereo(stereo_mod.Mode.OFF, 1.5)
            for _ in range(3):
                app.processEvents()
            view.grab()                       # a mono frame in the buffer
            view.set_stereo(stereo_mod.Mode.SIDE_BY_SIDE, 1.5)
            for _ in range(4):
                app.processEvents()
            image = view.grab().toImage()

            ratio = view.devicePixelRatioF()
            w = max(1, int(view.width() * ratio))
            h = max(1, int(view.height() * ratio))
            oy = view._stereo_layout(w, h)[1]
            assert oy > 4, "this structure leaves no margin to check"
            ground = tuple(round(255 * c) for c in view.theme.background)
            band = _row_colours(image, int(oy) // 2)
            assert band == {ground}, (
                f"the margin above the pair is {sorted(band)[:4]}, "
                f"not the background {ground}")
        finally:
            view.set_stereo(stereo_mod.Mode.OFF, 1.5)
            for _ in range(2):
                app.processEvents()

    def test_the_eye_renders_leave_the_context_pointing_elsewhere(self, view):
        """And paintGL has to put it back before it paints anything.

        A QOpenGLWidget's own framebuffer is not framebuffer 0 -- here it is 1 --
        and rendering the eyes to images binds buffers of its own and leaves
        them bound. Everything drawn afterwards goes wherever the binding
        points, so the overlay, and the blit of the pair itself, went to a
        surface that is never shown. The consequence on screen depends on the
        window and the driver, which is why this asserts the state rather than
        the pixels.

        Only asks for GL_FRAMEBUFFER_BINDING. glGetIntegerv(GL_VIEWPORT) takes
        the process down in PySide6 -- it returns four integers into a buffer
        sized for one.
        """
        if view._renderer is None:
            pytest.skip("no GL renderer on this tier")
        GL_FRAMEBUFFER_BINDING = 0x8CA6

        view.makeCurrent()
        try:
            gl = view._renderer.gl
            default = view.defaultFramebufferObject()
            assert gl.glGetIntegerv(GL_FRAMEBUFFER_BINDING) == default

            view.stereo_pair(120, 90)
            assert gl.glGetIntegerv(GL_FRAMEBUFFER_BINDING) != default, (
                "rendering the eyes no longer disturbs the binding -- if that "
                "is deliberate, this test and the restore it guards can go")

            view._renderer.bind_target(default, view.width(), view.height())
            assert gl.glGetIntegerv(GL_FRAMEBUFFER_BINDING) == default
        finally:
            view.doneCurrent()

    def test_a_side_by_side_frame_shows_two_separate_pictures(self, view, app):
        """Two pictures, one per pane, with the seam clear between them.

        Rendering the eyes binds framebuffers of its own at the size of one eye
        and leaves them bound. Everything after that is drawn with a QPainter,
        which goes to whatever framebuffer is bound through whatever viewport is
        set -- so a pair composed correctly at 814 by 302 was drawn as one eye
        stretched across the middle of the widget. The composition was right and
        the blit was right; only the state they landed in was wrong, which is
        why this looks at the widget's own pixels rather than at either.
        """
        from facet.core import theme as theme_mod
        from facet.gl import labels as labels_mod
        from facet.gl import stereo as stereo_mod

        before = (view.theme, view.labels.atom, view.show_axis_gizmo,
                  view.stereo)
        try:
            view.theme = theme_mod.Theme()
            view.labels.atom = labels_mod.AtomLabel.NONE
            view.show_axis_gizmo = False
            view.set_stereo(stereo_mod.Mode.SIDE_BY_SIDE, 1.5)
            for _ in range(4):
                app.processEvents()
            image = view.grab().toImage()
            # while the mode is still set: the layout is a property of it
            ratio = image.width() / max(view.width(), 1)
            ox, _, panes, ew, eh, gap = view._stereo_layout(
                int(view.width() * ratio), int(view.height() * ratio))
        finally:
            view.theme, view.labels.atom = before[0], before[1]
            view.show_axis_gizmo = before[2]
            view.set_stereo(before[3], 1.5)
            for _ in range(2):
                app.processEvents()

        rgb = _rgb_array(image).astype(int)
        ink = (rgb.sum(axis=2) < 3 * 235).mean(axis=0)
        drawn = np.flatnonzero(ink > 0.004)
        assert len(drawn) > 20, "nothing was drawn"
        seam = slice(int(ox + ew), int(ox + ew + gap) + 1)
        assert not (ink[seam] > 0.004).any(), (
            "the seam between the two pictures has something drawn in it, so "
            "what is on screen is not two pictures")
        left = drawn[drawn < ox + ew]
        right = drawn[drawn > ox + ew + gap]
        assert len(left) > 10 and len(right) > 10, (
            f"{len(left)} columns of ink in the left pane and {len(right)} in "
            f"the right: the widget is not showing a pair")

    def test_an_exported_image_carries_the_labels(self, view, app):
        """Turn labels on, export, and they were not there.

        The renderers draw geometry; everything written on the picture -- the
        labels, the axis gizmo, a measurement -- is painted over the top in
        paintGL, and the export path never did it. The SVG of the same view had
        all of them, so the two exports of one view disagreed.
        """
        from facet.gl import labels as labels_mod

        before = view.labels.atom
        try:
            view.labels.atom = labels_mod.AtomLabel.NONE
            plain = view.grab_image(500, 380, supersample=1)
            view.labels.atom = labels_mod.AtomLabel.SITE
            labelled = view.grab_image(500, 380, supersample=1)
        finally:
            view.labels.atom = before

        assert not plain.isNull() and not labelled.isNull()
        assert _difference(plain, labelled) > 200, (
            "the exported image is the same with labels on as with them off")

    def test_exported_text_is_not_subpixel_antialiased(self, view, app):
        """A figure is not read on the monitor it was drawn on.

        Qt draws text with the display's subpixel antialiasing on premultiplied
        images, and the renderers hand back premultiplied images -- so every
        label carried an orange fringe on one edge and a blue one on the other.
        On screen that is invisible; in print, in a slide, on someone else's
        monitor, it is a coloured halo.
        """
        from facet.core import theme as theme_mod
        from facet.gl import labels as labels_mod

        # The status line: text on a plain ground at the bottom left, with
        # nothing drawn over it. Atom labels sit on coloured atoms, where a
        # coloured pixel says nothing about how the text was rendered.
        before = (view.labels.atom, view.theme, view._status,
                  view.show_axis_gizmo)
        try:
            view.theme = theme_mod.Theme()          # white, so text is dark
            view.labels.atom = labels_mod.AtomLabel.NONE
            view.show_axis_gizmo = False
            view._status = "Bi1-O2  2.1834 A"
            image = view.grab_image(600, 460, supersample=1)
        finally:
            (view.labels.atom, view.theme, view._status,
             view.show_axis_gizmo) = before

        rgb = _rgb_array(image).astype(int)
        strip = rgb[int(0.93 * rgb.shape[0]):, :int(0.45 * rgb.shape[1])]
        ink = strip[strip.sum(axis=2) < 3 * 200]
        assert len(ink) > 60, (
            f"only {len(ink)} dark pixels where the status line should be")
        fringe = np.abs(ink[:, 0] - ink[:, 2])
        assert fringe.max() <= 40, (
            f"the text is colour-fringed: |R-B| up to {fringe.max()}")

    def test_rendering_a_pair_does_not_release_the_context(self, view, app):
        """The mechanism, stated directly rather than through its symptom."""
        from PySide6.QtGui import QOpenGLContext

        view.makeCurrent()
        try:
            assert QOpenGLContext.currentContext() is view.context()
            view._render_to_image(view.camera, 120, 100, 1)
            assert QOpenGLContext.currentContext() is view.context(), (
                "_render_to_image released a context it did not make current")
        finally:
            view.doneCurrent()

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
