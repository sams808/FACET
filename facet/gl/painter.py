"""The no-OpenGL fallback renderer.

Pure QPainter, painter's algorithm: sort everything by depth and draw back to
front. It runs anywhere Qt runs -- over remote desktop, on a locked-down
machine, in a virtual machine with no graphics driver at all.

It is not a placeholder. A radial gradient on a filled circle reads convincingly
as a lit sphere, Qt's antialiasing gives cleaner edges than a multisampled GL
context does, and depth cueing survives intact. What is lost is ambient
occlusion, properly ordered transparency, and the frame rate.

Two things the naive version of this gets wrong, both handled here:

* **Perspective scaling.** A sphere's screen radius falls off as 1/depth. Using
  one scale for the whole scene makes distant atoms too large, which reads as
  the model being inside-out.
* **Bond/atom interpenetration.** A per-primitive depth sort draws a bond
  entirely in front of or behind a sphere it actually passes through. Bonds are
  split at the midpoint and sorted in halves, so each half sorts against the
  atom it belongs to.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QRadialGradient

from .camera import Camera
from .scene import Scene


def _qcolor(rgb, alpha: float = 1.0) -> QColor:
    c = QColor()
    c.setRgbF(float(np.clip(rgb[0], 0, 1)),
              float(np.clip(rgb[1], 0, 1)),
              float(np.clip(rgb[2], 0, 1)),
              float(np.clip(alpha, 0, 1)))
    return c


class PainterRenderer:
    """Draws a scene with QPainter. Holds no state between frames."""

    background = QColor(22, 24, 28)

    def __init__(self):
        self.depth_cue = True
        self.fog_amount = 0.55
        self._scale = 1.0            # pixels per angstrom at the target plane
        self._distance = 1.0
        self._perspective = True
        self._near = 0.0
        self._far = 1.0

    # -- entry point -------------------------------------------------------
    def render(self, painter: QPainter, scene: Scene, camera: Camera,
               width: int, height: int, selected: int | None = None) -> None:
        painter.fillRect(0, 0, width, height, self.background)
        if scene is None or scene.n_atoms == 0:
            return

        self._scale = (height * 0.5) / max(camera.half_height(), 1e-6)
        self._distance = max(camera.distance, 1e-6)
        self._perspective = not camera.orthographic

        atoms = camera.project(scene.atom_position, width, height)
        self._near, self._far = self._depth_range(scene, camera, width, height)

        items: list[tuple[float, object]] = []
        self._collect_cell(items, scene, camera, width, height)
        self._collect_bonds(items, scene, camera, width, height)
        self._collect_atoms(items, scene, atoms, width, height)

        items.sort(key=lambda t: t[0])              # back to front
        painter.setRenderHint(QPainter.Antialiasing, True)
        for _, draw in items:
            draw(painter)

        if selected is not None and selected < scene.n_atoms:
            self._draw_selection(painter, scene, atoms, selected)

    # -- scaling and shading ----------------------------------------------
    def radius_at(self, world_radius: float, view_z: float) -> float:
        """Screen radius of a sphere, correct under either projection."""
        if not self._perspective:
            return float(world_radius) * self._scale
        depth = max(abs(float(view_z)), 1e-6)
        return float(world_radius) * self._scale * (self._distance / depth)

    def _fog(self, view_z: float) -> float:
        """How far this depth is through the scene, 0 near to 1 far."""
        if not self.depth_cue or self._far <= self._near:
            return 0.0
        t = (abs(float(view_z)) - self._near) / (self._far - self._near)
        return float(np.clip(t, 0.0, 1.0))

    def _cue(self, colour: QColor, view_z: float) -> QColor:
        """Blend a colour towards the background with depth."""
        t = self._fog(view_z) * self.fog_amount
        if t <= 0.0:
            return colour
        bg = self.background
        out = QColor()
        out.setRgbF(colour.redF() * (1 - t) + bg.redF() * t,
                    colour.greenF() * (1 - t) + bg.greenF() * t,
                    colour.blueF() * (1 - t) + bg.blueF() * t)
        return out

    def _depth_range(self, scene: Scene, camera: Camera,
                     width: int, height: int) -> tuple[float, float]:
        z = camera.project(scene.atom_position, width, height)[:, 2]
        z = np.abs(z[np.isfinite(z)])
        if z.size == 0:
            return 0.0, 1.0
        lo, hi = float(z.min()), float(z.max())
        return (lo, hi) if hi > lo else (lo, lo + 1.0)

    # -- collection --------------------------------------------------------
    def _collect_atoms(self, items, scene: Scene, projected: np.ndarray,
                       width: int, height: int) -> None:
        for i in range(scene.n_atoms):
            x, y, z = projected[i]
            if z >= 0:
                continue
            r = self.radius_at(float(scene.atom_radius[i]), z)
            if r < 0.4 or x + r < 0 or x - r > width or y + r < 0 or y - r > height:
                continue
            items.append((z, self._atom_drawer(x, y, r, scene.atom_color[i], z)))

    def _atom_drawer(self, x, y, r, rgb, view_z):
        def draw(painter: QPainter):
            shade = self._cue(_qcolor(rgb), view_z)
            # the highlight sits up and to the left, matching the GL path's light
            grad = QRadialGradient(QPointF(x - r * 0.35, y - r * 0.35), r * 1.5)
            grad.setColorAt(0.0, shade.lighter(165))
            grad.setColorAt(0.45, shade)
            grad.setColorAt(1.0, shade.darker(190))
            painter.setBrush(QBrush(grad))
            painter.setPen(QPen(shade.darker(260), max(0.6, r * 0.06)))
            painter.drawEllipse(QPointF(x, y), r, r)
        return draw

    def _collect_bonds(self, items, scene: Scene, camera: Camera,
                       width: int, height: int) -> None:
        if scene.n_bonds == 0:
            return
        mid = (scene.bond_a + scene.bond_b) * 0.5
        pa = camera.project(scene.bond_a, width, height)
        pb = camera.project(scene.bond_b, width, height)
        pm = camera.project(mid, width, height)

        for i in range(scene.n_bonds):
            world_w = float(scene.bond_radius[i]) * 2.0
            for p, q, colour in ((pa[i], pm[i], scene.bond_color_a[i]),
                                 (pm[i], pb[i], scene.bond_color_b[i])):
                if p[2] >= 0 or q[2] >= 0:
                    continue
                depth = (p[2] + q[2]) * 0.5
                w = max(0.7, self.radius_at(world_w, depth))
                items.append((depth, self._bond_drawer(p, q, w, colour, depth)))

    def _bond_drawer(self, p, q, w, rgb, view_z):
        def draw(painter: QPainter):
            painter.setPen(QPen(self._cue(_qcolor(rgb), view_z), w,
                                Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(QPointF(p[0], p[1]), QPointF(q[0], q[1]))
        return draw

    def _collect_cell(self, items, scene: Scene, camera: Camera,
                      width: int, height: int) -> None:
        if len(scene.cell_segments) == 0:
            return
        flat = scene.cell_segments.reshape(-1, 3)
        px = camera.project(flat, width, height).reshape(-1, 2, 3)
        for seg in px:
            if seg[0, 2] >= 0 or seg[1, 2] >= 0:
                continue
            # the cell outline is a reference frame, not an object: always
            # behind, so it never cuts across an atom
            items.append((-np.inf, self._cell_drawer(seg)))

    def _cell_drawer(self, seg):
        def draw(painter: QPainter):
            painter.setPen(QPen(QColor(110, 118, 134), 1.0))
            painter.drawLine(QPointF(seg[0, 0], seg[0, 1]),
                             QPointF(seg[1, 0], seg[1, 1]))
        return draw

    # -- overlay -----------------------------------------------------------
    def _draw_selection(self, painter: QPainter, scene: Scene,
                        projected: np.ndarray, index: int) -> None:
        x, y, z = projected[index]
        if z >= 0:
            return
        r = max(4.0, self.radius_at(float(scene.atom_radius[index]), z))
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(255, 214, 92), 2.0))
        painter.drawEllipse(QPointF(x, y), r + 4, r + 4)
