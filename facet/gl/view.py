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
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget

from . import caps as caps_mod
from .camera import Camera
from .scene import Scene, Style


class StructureView(QOpenGLWidget):
    """Interactive 3D view of a structure."""

    atomPicked = Signal(int)            # atom index, or -1 for empty space
    sitePicked = Signal(int)            # site index, or -1
    measured = Signal(str)              # human-readable measurement
    ready = Signal(object)              # Capabilities, once the context exists

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        self.camera = Camera()
        self.scene: Scene | None = None
        self.caps: caps_mod.Capabilities | None = None

        self._renderer = None
        self._gl = None
        self._failed = False
        self._fallback = None

        self._last_pos: QPoint | None = None
        self._button = Qt.NoButton
        self._selected: int | None = None
        self._measure_chain: list[int] = []
        self._show_labels = False
        self._label_elements_only = True
        self._status = ""

    # -- public API --------------------------------------------------------
    def set_scene(self, scene: Scene, reframe: bool = True) -> None:
        self.scene = scene
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

    def set_labels(self, on: bool, elements_only: bool = True) -> None:
        self._show_labels = bool(on)
        self._label_elements_only = bool(elements_only)
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
        if self._renderer is None:
            return self.grab().toImage()
        w = int(width or self.width())
        h = int(height or self.height())
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
        if self.caps is not None:
            self.caps.tier = caps_mod.Tier.BASIC
            self.caps.reason = reason

    def resizeGL(self, w: int, h: int) -> None:
        if self._renderer is not None:
            self._renderer.resize(w, h)

    def paintGL(self) -> None:
        if self._renderer is not None and self.scene is not None:
            ratio = self.devicePixelRatioF()
            w = max(1, int(self.width() * ratio))
            h = max(1, int(self.height() * ratio))
            self._renderer.target_fbo = self.defaultFramebufferObject()
            try:
                self._renderer.render(self.camera, w, h)
            finally:
                self._renderer.target_fbo = None

    def paintEvent(self, event) -> None:
        # QOpenGLWidget runs paintGL from inside paintEvent; calling the base
        # first draws the GL content, then the QPainter overlay goes on top.
        if self._renderer is not None:
            super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        if self._renderer is None:
            self._paint_fallback(painter)
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
        if self._show_labels:
            self._paint_labels(painter)
        if self._selected is not None:
            self._paint_selection(painter)
        if len(self._measure_chain) >= 2:
            self._paint_measurement(painter)
        if self._status:
            self._paint_status(painter)

    def _paint_placeholder(self, painter: QPainter) -> None:
        painter.fillRect(self.rect(), QColor(22, 24, 28))
        painter.setPen(QColor(150, 155, 165))
        f = QFont(painter.font())
        f.setPointSizeF(f.pointSizeF() + 1)
        painter.setFont(f)
        painter.drawText(self.rect(), Qt.AlignCenter,
                         "Open a structure, or drop a .cif file here")

    def _visible(self, positions: np.ndarray) -> np.ndarray:
        return self.camera.project(positions, self.width(), self.height())

    def _paint_labels(self, painter: QPainter) -> None:
        s = self.scene
        if s.n_atoms == 0:
            return
        px = self._visible(s.atom_position)
        f = QFont(painter.font())
        f.setPointSizeF(max(7.5, f.pointSizeF() - 0.5))
        painter.setFont(f)
        painter.setPen(QColor(235, 238, 245))

        order = np.argsort(px[:, 2])[::-1]     # far to near
        drawn: list[tuple[float, float]] = []
        for i in order:
            x, y, z = px[i]
            if z >= 0 or not (0 <= x <= self.width() and 0 <= y <= self.height()):
                continue
            # cheap de-clutter: skip a label that would sit on another
            if any((x - dx) ** 2 + (y - dy) ** 2 < 400 for dx, dy in drawn):
                continue
            drawn.append((x, y))
            text = (s.atom_element[i] if self._label_elements_only
                    else s.atom_label[i])
            painter.drawText(QPoint(int(x) + 6, int(y) - 6), text)

    def _paint_selection(self, painter: QPainter) -> None:
        s = self.scene
        i = self._selected
        if i is None or i >= s.n_atoms:
            return
        px = self._visible(s.atom_position[i:i + 1])[0]
        if px[2] >= 0:
            return
        r = self._screen_radius(s.atom_position[i], s.atom_radius[i])
        pen = QPen(QColor(255, 214, 92), 2.0)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPoint(int(px[0]), int(px[1])),
                            int(r + 4), int(r + 4))
        painter.setPen(QColor(255, 214, 92))
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
        for p in px:
            painter.drawEllipse(QPoint(int(p[0]), int(p[1])), 3, 3)

    def _paint_status(self, painter: QPainter) -> None:
        painter.setPen(QColor(210, 216, 226))
        rect = self.rect().adjusted(10, 0, -10, -8)
        painter.drawText(rect, Qt.AlignLeft | Qt.AlignBottom, self._status)

    # -- interaction -------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        self._last_pos = event.position().toPoint()
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
        self._last_pos = pos

    def mouseReleaseEvent(self, event) -> None:
        moved = False
        if self._last_pos is not None:
            start = event.position().toPoint()
            moved = (abs(start.x() - self._last_pos.x()) > 2
                     or abs(start.y() - self._last_pos.y()) > 2)
        if (event.button() == Qt.LeftButton and not moved
                and not (event.modifiers() & Qt.ShiftModifier)):
            self._handle_select_click(event.position().toPoint())
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
            self.set_labels(not self._show_labels, self._label_elements_only)
        elif key == Qt.Key_O:
            self.set_projection(not self.camera.orthographic)
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
