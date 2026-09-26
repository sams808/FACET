"""The 3D viewport widget.

A ``QOpenGLWidget`` that drives :class:`~facet.gl.renderer.Renderer`, plus a
``QPainter`` overlay for labels and measurements. Text drawn with QPainter stays
crisp at any device pixel ratio and needs no glyph atlas, which is worth more
here than drawing it in GL would be.

If no usable OpenGL context can be created the widget falls back to
:mod:`facet.gl.painter`, a pure-Qt renderer. The application must open on a
machine with no graphics hardware, and a viewport that refuses to appear is
worse than one that draws flatly.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPen
from PySide6.QtOpenGL import QOpenGLPaintDevice
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget

from . import caps as caps_mod
from . import stereo as stereo_mod
from . import labels as labels_mod
from .camera import Camera
from .scene import Scene, Style


class StructureView(QOpenGLWidget):
    """Interactive 3D view of a structure."""

    atomPicked = Signal(int)            # atom index, or -1 for empty space
    sitePicked = Signal(int)            # site index, or -1
    measured = Signal(str)              # human-readable measurement
    ready = Signal(object)              # Capabilities, once the context exists
    labelsChanged = Signal(object)      # LabelSettings, after a keyboard cycle
    contextRequested = Signal(int, object)   # atom index or -1, and the position

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        # The right button pans, so Qt must not open a menu on press. The menu
        # is raised from mouseReleaseEvent instead, and only when the pointer
        # did not travel -- a pan and a context click are the same gesture up
        # until the pointer moves.
        self.setContextMenuPolicy(Qt.PreventContextMenu)

        self.camera = Camera()
        self.scene: Scene | None = None
        # Stereo is off by default and costs nothing while it is: the second
        # eye is only rendered when a mode asks for it.
        self.stereo = stereo_mod.Mode.OFF
        self.stereo_separation = Camera.STEREO_SEPARATION
        self.caps: caps_mod.Capabilities | None = None
        self.theme = None

        self._renderer = None
        self._gl = None
        self._failed = False
        self._fallback = None

        self._last_pos: QPoint | None = None
        # Where the button went down, and how far the pointer has travelled
        # since. Kept separately from _last_pos, which follows the pointer: a
        # release compared against _last_pos is comparing a point with itself,
        # so every drag looked like a click and rotating the view selected
        # whatever atom happened to be under the cursor when the button came up.
        self._press_pos: QPoint | None = None
        self._travel = 0.0
        self._button = Qt.NoButton
        self._selected: int | None = None
        self._measure_chain: list[int] = []
        self._status = ""
        self.labels = labels_mod.LabelSettings()
        self.structure = None
        self.results = None
        self.show_axis_gizmo = True

    # -- public API --------------------------------------------------------
    def set_theme(self, theme) -> None:
        """Adopt a theme. The scene must be rebuilt separately for the colour
        mode and scales, which are baked into its vertex arrays."""
        self.theme = theme
        if self._renderer is not None:
            self.makeCurrent()
            try:
                self._renderer.apply_theme(theme)
            finally:
                self.doneCurrent()
        if self._fallback is not None:
            self._fallback.apply_theme(theme)
        self.update()

    def set_scene(self, scene: Scene, reframe: bool = True) -> None:
        self.scene = scene
        if scene is not None and scene.theme is not None:
            self.theme = scene.theme
        if reframe:
            self.camera.frame(scene.center, scene.radius)
        self._selected = None
        self._measure_chain.clear()
        if self._renderer is not None:
            self.makeCurrent()
            try:
                self._renderer.set_scene(scene)
            finally:
                self.doneCurrent()
        self.update()

    def set_threshold(self, v_bond: float) -> None:
        """Move the bond threshold.

        Restyles the cached scene and re-uploads only the bond geometry. No
        neighbour search, no analysis: this is the interaction the whole
        architecture was arranged around, and it must stay free.
        """
        if self.scene is None:
            return
        self.scene.restyle(v_bond)
        if self._renderer is not None:
            self.makeCurrent()
            try:
                self._renderer.update_bond_colors(self.scene)
            finally:
                self.doneCurrent()
        self.update()

    def set_labels(self, settings: labels_mod.LabelSettings) -> None:
        """Adopt a whole label configuration."""
        self.labels = settings
        self.update()

    def set_label_context(self, structure=None, results=None) -> None:
        """Supply what the labels need beyond the scene.

        Wyckoff letters, oxidation states, coordination numbers and phi live on
        the structure and its analysis, not on the drawable scene, so labelling
        by them needs both.
        """
        self.structure = structure
        self.results = results
        self.update()

    def set_projection(self, orthographic: bool) -> None:
        self.camera.orthographic = bool(orthographic)
        self.update()

    def set_field_of_view(self, degrees: float) -> None:
        self.camera.set_fov(degrees)
        self.update()

    def reset_view(self) -> None:
        if self.scene is not None:
            self.camera.reset_orientation()
            self.camera.frame(self.scene.center, self.scene.radius)
        self.update()

    def view_along(self, axis) -> None:
        self.camera.view_along(axis)
        self.update()

    def select_atom(self, index: int | None) -> None:
        self._selected = index
        self.update()

    def set_stereo(self, mode, separation: float | None = None) -> None:
        """Choose a stereo mode. Off means one render, as before."""
        self.stereo = mode
        if separation is not None:
            self.stereo_separation = float(separation)
        self.update()

    def stereo_pair(self, width: int, height: int, supersample: int = 1):
        """Render both eyes and return ``(left, right)`` as QImages.

        Nothing about stereo reaches the renderers: each eye is just a different
        camera, which is why this works the same on all three tiers.
        """
        left = self.camera.for_eye(-1, self.stereo_separation)
        right = self.camera.for_eye(+1, self.stereo_separation)
        return (self._render_to_image(left, width, height, supersample),
                self._render_to_image(right, width, height, supersample))

    def _render_to_image(self, camera, width: int, height: int,
                         supersample: int = 1):
        """One camera to a QImage, on whichever tier is in use."""
        from PySide6.QtGui import QImage, QPainter

        if self._renderer is not None:
            self.makeCurrent()
            try:
                previous = self._renderer.target_fbo
                self._renderer.target_fbo = None
                try:
                    return self._renderer.to_image(camera, width, height,
                                                   supersample)
                finally:
                    self._renderer.target_fbo = previous
            finally:
                self.doneCurrent()

        image = QImage(width, height, QImage.Format_RGBA8888)
        image.fill(0)
        painter = QPainter(image)
        try:
            if self._fallback is not None and self.scene is not None:
                self._fallback.render(painter, self.scene, camera,
                                      width, height, selected=self._selected)
        finally:
            painter.end()
        return image

    def clear_measurement(self) -> None:
        self._measure_chain.clear()
        self._status = ""
        self.update()

    def grab_image(self, width: int | None = None, height: int | None = None,
                   supersample: int = 3):
        """Render at high resolution for export.

        Supersampled and downsampled rather than multisampled: this driver
        declines MSAA on the default framebuffer, and a 3x downsample is what a
        600 dpi figure wants regardless.
        """
        w = int(width or self.width())
        h = int(height or self.height())
        if self.stereo.needs_two_eyes and self.scene is not None:
            left, right = self.stereo_pair(w, h, supersample)
            return stereo_mod.combine(left, right, self.stereo)
        if self._renderer is None:
            # Render at the size asked for rather than grabbing the widget.
            # Grabbing ignores width and height entirely, so on a machine with
            # no graphics card "export at 2000 px" quietly produced a picture
            # the size of the window -- the one tier where a user is most likely
            # to need a bigger image than the screen.
            if self.scene is not None:
                return self._render_to_image(self.camera, w, h)
            return self.grab().toImage()
        self.makeCurrent()
        try:
            previous, self._renderer.target_fbo = self._renderer.target_fbo, None
            image = self._renderer.to_image(self.camera, w, h, supersample)
            self._renderer.target_fbo = previous
        finally:
            self.doneCurrent()
        return image

    # -- GL lifecycle ------------------------------------------------------
    def initializeGL(self) -> None:
        from .renderer import Renderer, ShaderError

        context = self.context()
        self.caps = caps_mod.detect(context)

        if self.caps.tier is caps_mod.Tier.BASIC:
            self._enter_fallback(self.caps.reason or "no usable OpenGL")
            self.ready.emit(self.caps)
            return

        self._gl, _ = caps_mod._load_functions(self.caps.major, self.caps.minor)
        try:
            self._renderer = Renderer(self._gl, self.caps)
            if self.theme is not None:
                self._renderer.apply_theme(self.theme)
            if self.scene is not None:
                self._renderer.set_scene(self.scene)
        except (ShaderError, Exception) as exc:      # noqa: B014 - deliberate
            self._enter_fallback(f"shader setup failed: {exc}")
        self.ready.emit(self.caps)

    def _enter_fallback(self, reason: str) -> None:
        from .painter import PainterRenderer

        self._failed = True
        self._renderer = None
        self._fallback = PainterRenderer()
        if self.theme is not None:
            self._fallback.apply_theme(self.theme)
        if self.caps is not None:
            self.caps.tier = caps_mod.Tier.BASIC
            self.caps.reason = reason

    def resizeGL(self, w: int, h: int) -> None:
        if self._renderer is not None:
            self._renderer.resize(w, h)

    def paintGL(self) -> None:
        """GL first, then the label and measurement overlay.

        The overlay is painted onto a **QOpenGLPaintDevice**, not onto the
        widget. This is not a stylistic choice, and it cost an afternoon:

        * A QPainter opened on the widget from ``paintEvent`` targets a surface
          that is discarded. Nothing appears, and no error is raised.
        * A QPainter opened on the widget from ``paintGL`` works only while the
          renderer has not bound a framebuffer of its own. FACET composites
          straight into the widget's framebuffer, which leaves Qt's own
          tracking of the bound framebuffer stale, and QPainter then draws into
          nothing -- again silently.
        * ``QOpenGLFramebufferObject.bindDefault()`` does not rescue it: it
          binds framebuffer 0, and a QOpenGLWidget's default framebuffer is its
          own, not 0.

        A QOpenGLPaintDevice draws into whatever framebuffer is currently
        bound, which is exactly the one the GL output went to. It is the
        supported way to paint over a custom framebuffer render.
        """
        ratio = self.devicePixelRatioF()
        w = max(1, int(self.width() * ratio))
        h = max(1, int(self.height() * ratio))

        stereo_image = None
        if self.stereo.needs_two_eyes and self.scene is not None:
            # Both eyes are rendered offscreen and combined on the pixels, then
            # the result is blitted with QPainter. Doing it that way rather than
            # with colour masks in the shader is what lets the software tier and
            # the image export use exactly the same stereo code.
            left, right = self.stereo_pair(w, h)
            stereo_image = stereo_mod.combine(left, right, self.stereo)
        elif self._renderer is not None and self.scene is not None:
            self._renderer.target_fbo = self.defaultFramebufferObject()
            try:
                self._renderer.render(self.camera, w, h)
            finally:
                self._renderer.target_fbo = None
            # and hand the context back clean, or the glyphs fail the depth
            # test left over from the geometry pass
            self._renderer.reset_state()

        if self._renderer is not None:
            device = QOpenGLPaintDevice(QSize(w, h))
            device.setDevicePixelRatio(ratio)
            painter = QPainter(device)
        else:
            painter = QPainter(self)

        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        if stereo_image is not None:
            painter.drawImage(0, 0, stereo_image.scaled(
                w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        elif self._renderer is None:
            self._paint_fallback(painter)
        # The overlay -- labels, the gizmo, a measurement -- is drawn once, from
        # the centre view. Drawing it per eye would put the text at two different
        # depths and it would not fuse; a single overlay reads as floating in
        # front, which is where an annotation belongs.
        self._paint_overlay(painter)
        painter.end()

    def _paint_fallback(self, painter: QPainter) -> None:
        if self._fallback is None or self.scene is None:
            painter.fillRect(self.rect(), QColor(22, 24, 28))
            return
        self._fallback.render(painter, self.scene, self.camera,
                              self.width(), self.height(),
                              selected=self._selected)

    # -- overlay -----------------------------------------------------------
    def _paint_overlay(self, painter: QPainter) -> None:
        if self.scene is None:
            self._paint_placeholder(painter)
            return
        self._paint_labels(painter)
        if self._selected is not None:
            self._paint_selection(painter)
        if len(self._measure_chain) >= 2:
            self._paint_measurement(painter)
        if self.show_axis_gizmo and self.structure is not None:
            self._paint_axis_gizmo(painter)
        if self._status:
            self._paint_status(painter)

    def _paint_placeholder(self, painter: QPainter) -> None:
        painter.fillRect(self.rect(), self._ink(self.theme.background
                                                if self.theme
                                                else (0.086, 0.094, 0.110)))
        painter.setPen(QColor(150, 155, 165))
        f = QFont(painter.font())
        f.setPointSizeF(f.pointSizeF() + 1)
        painter.setFont(f)
        painter.drawText(self.rect(), Qt.AlignCenter,
                         "Open a structure, or drop a .cif file here")

    def _visible(self, positions: np.ndarray) -> np.ndarray:
        return self.camera.project(positions, self.width(), self.height())

    def _paint_labels(self, painter: QPainter) -> None:
        """Atom, bond, axis and measurement labels, de-cluttered and haloed."""
        s = self.scene
        settings = self.labels
        if s is None or s.n_atoms == 0:
            return
        if not settings.any_enabled() and not (
                settings.show_measurements and len(self._measure_chain) >= 2):
            return

        font = QFont(painter.font())
        font.setPointSizeF(max(6.0, settings.font_points))
        font.setBold(settings.bold)
        painter.setFont(font)
        metrics = QFontMetricsF(font)

        def measure(text: str) -> tuple[float, float]:
            return metrics.horizontalAdvance(text), metrics.height()

        placed = labels_mod.build(
            s, self.camera, settings, width=self.width(), height=self.height(),
            measure=measure, results=self.results, structure=self.structure,
            selected_site=self._selected_site())

        if settings.show_measurements and len(self._measure_chain) >= 2:
            placed += labels_mod.measurement_labels(
                s, self.camera, self._measure_chain,
                width=self.width(), height=self.height(),
                decimals=settings.decimals)

        base = (settings.color if settings.color is not None
                else (self.theme.contrasting_ink() if self.theme
                      else (0.92, 0.93, 0.96)))
        ink = self._ink(base)
        halo = self._halo_color()

        for label in placed:
            colour = ink
            if label.kind == "measurement":
                colour = self._ink(self.theme.selection_color if self.theme
                                   else (1.0, 0.84, 0.36))
            elif label.kind == "bond" and not label.emphasis:
                colour = self._ink(self.theme.subthreshold_color if self.theme
                                   else (0.62, 0.64, 0.68))
            self._draw_haloed(painter, label.text, label.x, label.y,
                              colour, halo, settings.halo)

    def _paint_axis_gizmo(self, painter: QPainter) -> None:
        """a, b and c drawn in the corner, showing which way the cell points.

        Drawn from the camera's rotation only, with no translation and no
        perspective, so it reports orientation and nothing else. It sits in the
        corner rather than on the structure because it has to stay readable
        when the cell is off screen or zoomed past.
        """
        from .camera import quat_to_matrix

        size = 46
        margin = 14
        cx = self.width() - size - margin
        cy = self.height() - size - margin

        orth = self.structure.cell.orth
        rot = quat_to_matrix(self.camera.orientation)

        axes = []
        for i, name in enumerate(("a", "b", "c")):
            world = orth[:, i]
            n = np.linalg.norm(world)
            if n < 1e-9:
                continue
            view = rot @ (world / n)
            # screen x to the right, screen y downwards
            axes.append((name, float(view[0]), float(-view[1]), float(view[2])))

        # draw the axis pointing away from the viewer first, so the nearer
        # ones overlap it rather than the other way round
        axes.sort(key=lambda a: a[3])

        colours = {"a": QColor("#e8674f"), "b": QColor("#7bc96f"),
                   "c": QColor("#5b9bd5")}
        font = QFont(painter.font())
        font.setPointSizeF(8.5)
        font.setBold(True)
        painter.setFont(font)

        for name, dx, dy, dz in axes:
            # foreshorten with depth so the gizmo reads as three dimensional
            length = size * 0.78
            x2 = cx + dx * length
            y2 = cy + dy * length
            colour = QColor(colours[name])
            if dz < 0:
                colour.setAlpha(120)          # pointing away from the viewer
            painter.setPen(QPen(colour, 2.0, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(QPointF(cx, cy), QPointF(x2, y2))
            painter.setPen(colour)
            self._draw_haloed(painter, name, x2 + 3, y2 + 4,
                              colour, self._halo_color(), True)

        painter.setPen(QPen(QColor(150, 155, 165, 120), 1.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), 2.0, 2.0)

    def _halo_color(self) -> QColor:
        """A contrasting outline, so text reads over an atom, the background or
        a transparent polyhedron without having to know which."""
        if self.theme is not None and self.theme.is_light_background:
            return QColor(255, 255, 255, 215)
        return QColor(0, 0, 0, 205)

    @staticmethod
    def _draw_haloed(painter: QPainter, text: str, x: float, y: float,
                     colour: QColor, halo: QColor, enabled: bool) -> None:
        if enabled:
            painter.setPen(halo)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (1, -1), (-1, 1), (1, 1)):
                painter.drawText(QPointF(x + dx, y + dy), text)
        painter.setPen(colour)
        painter.drawText(QPointF(x, y), text)

    def _selected_site(self) -> int | None:
        if self._selected is None or self.scene is None:
            return None
        if self._selected >= self.scene.n_atoms:
            return None
        return int(self.scene.atom_site[self._selected])

    def _paint_selection(self, painter: QPainter) -> None:
        s = self.scene
        i = self._selected
        if i is None or i >= s.n_atoms:
            return
        px = self._visible(s.atom_position[i:i + 1])[0]
        if px[2] >= 0:
            return
        r = self._screen_radius(s.atom_position[i], s.atom_radius[i])
        pen = QPen(self._ink(self.theme.selection_color if self.theme
                            else (1.0, 0.84, 0.36)), 2.0)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPoint(int(px[0]), int(px[1])),
                            int(r + 4), int(r + 4))
        painter.setPen(self._ink(self.theme.selection_color if self.theme
                                 else (1.0, 0.84, 0.36)))
        painter.drawText(QPoint(int(px[0]) + int(r) + 8, int(px[1]) - int(r) - 2),
                         s.atom_label[i])

    def _screen_radius(self, position, radius) -> float:
        """Radius in pixels of a sphere at a world position."""
        centre = self.camera.project([position], self.width(), self.height())[0]
        offset = np.array(position, float) + self.camera_right() * float(radius)
        edge = self.camera.project([offset], self.width(), self.height())[0]
        return float(np.hypot(edge[0] - centre[0], edge[1] - centre[1]))

    def camera_right(self) -> np.ndarray:
        from .camera import quat_to_matrix

        return quat_to_matrix(self.camera.orientation)[0, :3]

    def _paint_measurement(self, painter: QPainter) -> None:
        s = self.scene
        pts = [s.atom_position[i] for i in self._measure_chain if i < s.n_atoms]
        if len(pts) < 2:
            return
        px = self._visible(np.array(pts))
        pen = QPen(QColor(120, 220, 255), 1.6, Qt.DashLine)
        painter.setPen(pen)
        for a, b in zip(px[:-1], px[1:]):
            painter.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
        for point in px:
            painter.drawEllipse(QPoint(int(point[0]), int(point[1])), 3, 3)

    @staticmethod
    def _ink(rgb) -> QColor:
        c = QColor()
        c.setRgbF(float(rgb[0]), float(rgb[1]), float(rgb[2]))
        return c

    def _paint_status(self, painter: QPainter) -> None:
        painter.setPen(self._ink(self.theme.contrasting_ink() if self.theme
                                 else (0.82, 0.85, 0.89)))
        rect = self.rect().adjusted(10, 0, -10, -8)
        painter.drawText(rect, Qt.AlignLeft | Qt.AlignBottom, self._status)

    # -- interaction -------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        self._last_pos = event.position().toPoint()
        self._press_pos = self._last_pos
        self._travel = 0.0
        self._button = event.button()
        if event.button() == Qt.LeftButton and event.modifiers() & Qt.ShiftModifier:
            self._handle_measure_click(self._last_pos)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        if self._last_pos is None:
            self._last_pos = pos
            return
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()

        if self._button == Qt.LeftButton and not (event.modifiers() & Qt.ShiftModifier):
            self.camera.drag_rotate(self._last_pos.x(), self._last_pos.y(),
                                    pos.x(), pos.y(), self.width(), self.height())
            self.update()
        elif self._button in (Qt.MiddleButton, Qt.RightButton):
            self.camera.drag_pan(dx, dy, self.width(), self.height())
            self.update()
        self._travel += (dx * dx + dy * dy) ** 0.5
        self._last_pos = pos

    DRAG_TOLERANCE = 3.0        # pixels; beyond this a press was a drag

    def mouseReleaseEvent(self, event) -> None:
        position = event.position().toPoint()
        moved = self._travel > self.DRAG_TOLERANCE
        if self._press_pos is not None:
            straight = ((position.x() - self._press_pos.x()) ** 2
                        + (position.y() - self._press_pos.y()) ** 2) ** 0.5
            moved = moved or straight > self.DRAG_TOLERANCE

        if (event.button() == Qt.LeftButton and not moved
                and not (event.modifiers() & Qt.ShiftModifier)):
            self._handle_select_click(position)
        elif event.button() == Qt.RightButton and not moved:
            # A right-click that did not pan asks for the context menu. The menu
            # itself is the window's business: it acts on the project, not on
            # the view.
            picked = self.pick_at(position)
            self.contextRequested.emit(-1 if picked is None else int(picked),
                                       self.mapToGlobal(position))

        self._travel = 0.0
        self._press_pos = None
        self._button = Qt.NoButton
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        self.camera.zoom(event.angleDelta().y() / 120.0)
        self.update()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_R:
            self.reset_view()
        elif key == Qt.Key_L:
            # cycle through the atom label kinds, so L keeps working as the
            # quick "show me what these are" key
            kinds = list(labels_mod.AtomLabel)
            i = kinds.index(self.labels.atom)
            self.labels.atom = kinds[(i + 1) % len(kinds)]
            self.labelsChanged.emit(self.labels)
            self.update()
        elif key == Qt.Key_O:
            self.set_projection(not self.camera.orthographic)
        elif key == Qt.Key_G:
            self.show_axis_gizmo = not self.show_axis_gizmo
            self.update()
        elif key == Qt.Key_Escape:
            self.clear_measurement()
            self.select_atom(None)
        else:
            super().keyPressEvent(event)

    # -- picking -----------------------------------------------------------
    def pick_at(self, pos: QPoint) -> int | None:
        if self._renderer is None or self.scene is None:
            return self._pick_on_cpu(pos)
        ratio = self.devicePixelRatioF()
        self.makeCurrent()
        try:
            index = self._renderer.pick(
                self.camera,
                max(1, int(self.width() * ratio)),
                max(1, int(self.height() * ratio)),
                int(pos.x() * ratio), int(pos.y() * ratio))
        finally:
            self.doneCurrent()
        if index is None or not (0 <= index < self.scene.n_atoms):
            return None
        return int(index)

    def _pick_on_cpu(self, pos: QPoint) -> int | None:
        """Nearest projected atom, used by the QPainter fallback."""
        s = self.scene
        if s is None or s.n_atoms == 0:
            return None
        px = self._visible(s.atom_position)
        front = px[:, 2] < 0
        if not front.any():
            return None
        d = np.hypot(px[:, 0] - pos.x(), px[:, 1] - pos.y())
        d[~front] = np.inf
        i = int(np.argmin(d))
        r = self._screen_radius(s.atom_position[i], s.atom_radius[i])
        return i if d[i] <= max(r, 4.0) else None

    def _handle_select_click(self, pos: QPoint) -> None:
        index = self.pick_at(pos)
        self._selected = index
        if index is None:
            self.atomPicked.emit(-1)
            self.sitePicked.emit(-1)
        else:
            self.atomPicked.emit(index)
            self.sitePicked.emit(int(self.scene.atom_site[index]))
        self.update()

    def _handle_measure_click(self, pos: QPoint) -> None:
        index = self.pick_at(pos)
        if index is None:
            return
        self._measure_chain.append(index)
        if len(self._measure_chain) > 4:
            self._measure_chain = self._measure_chain[-4:]
        self._status = self._describe_measurement()
        if self._status:
            self.measured.emit(self._status)
        self.update()

    def _describe_measurement(self) -> str:
        s = self.scene
        pts = [s.atom_position[i] for i in self._measure_chain]
        names = [s.atom_label[i] for i in self._measure_chain]
        if len(pts) == 2:
            d = float(np.linalg.norm(pts[1] - pts[0]))
            return f"{names[0]}–{names[1]}  {d:.4f} Å"
        if len(pts) == 3:
            a = pts[0] - pts[1]
            b = pts[2] - pts[1]
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
            return f"{names[0]}–{names[1]}–{names[2]}  {ang:.2f}°"
        if len(pts) == 4:
            b0 = pts[0] - pts[1]
            b1 = pts[2] - pts[1]
            b2 = pts[3] - pts[2]
            n1 = np.cross(b0, b1)
            n2 = np.cross(b1, b2)
            m = np.cross(n1, b1 / np.linalg.norm(b1))
            x = float(np.dot(n1, n2))
            y = float(np.dot(m, n2))
            ang = np.degrees(np.arctan2(y, x))
            return (f"{names[0]}–{names[1]}–{names[2]}–{names[3]}"
                    f"  {ang:.2f}° torsion")
        return ""
