"""Camera and trackball.

Pure numpy: no Qt, no GL. That keeps the part most likely to be subtly wrong --
the matrices -- testable without a window.

Orientation is held as a quaternion rather than Euler angles. Accumulating
rotations as angles gives gimbal lock and, worse, makes the drag direction
depend on the current view, which feels broken to use. A quaternion composed
from an arcball has neither problem.

Matrix convention: row-major numpy acting on column vectors, i.e. ``M @ v``.
The GL layer transposes on upload.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# --- quaternions -------------------------------------------------------------

def quat_identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])          # w, x, y, z


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return quat_identity()
    axis = axis / n
    s = math.sin(angle / 2.0)
    return np.array([math.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _arcball_vector(x: float, y: float, width: int, height: int) -> np.ndarray:
    """Map a screen point onto Shoemake's sphere-plus-hyperbola.

    Inside the ball the point lifts onto a sphere; outside it follows a
    hyperbolic sheet, so a drag that leaves the ball keeps rotating smoothly
    instead of sticking at the rim.
    """
    r = min(width, height) * 0.5
    if r <= 0:
        return np.array([0.0, 0.0, 1.0])
    vx = (x - width * 0.5) / r
    vy = (height * 0.5 - y) / r
    d2 = vx * vx + vy * vy
    if d2 <= 0.5:
        vz = math.sqrt(1.0 - d2)
    else:
        vz = 0.5 / math.sqrt(d2)
    v = np.array([vx, vy, vz])
    return v / np.linalg.norm(v)


# --- matrices ----------------------------------------------------------------

def look_at(eye, target, up) -> np.ndarray:
    eye = np.asarray(eye, float)
    f = np.asarray(target, float) - eye
    n = np.linalg.norm(f)
    if n < 1e-12:
        f = np.array([0.0, 0.0, -1.0])
    else:
        f = f / n
    up = np.asarray(up, float)
    s = np.cross(f, up)
    ns = np.linalg.norm(s)
    if ns < 1e-9:                      # looking straight along up
        up = np.array([0.0, 0.0, 1.0]) if abs(f[1]) > 0.9 else np.array([0.0, 1.0, 0.0])
        s = np.cross(f, up)
        ns = np.linalg.norm(s)
    s = s / ns
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def perspective(fovy_deg: float, aspect: float, znear: float, zfar: float) -> np.ndarray:
    t = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4))
    m[0, 0] = t / max(aspect, 1e-6)
    m[1, 1] = t
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = 2.0 * zfar * znear / (znear - zfar)
    m[3, 2] = -1.0
    return m


def orthographic(half_height: float, aspect: float,
                 znear: float, zfar: float) -> np.ndarray:
    hh = max(half_height, 1e-6)
    hw = hh * max(aspect, 1e-6)
    m = np.eye(4)
    m[0, 0] = 1.0 / hw
    m[1, 1] = 1.0 / hh
    m[2, 2] = -2.0 / (zfar - znear)
    m[2, 3] = -(zfar + znear) / (zfar - znear)
    return m


# --- the camera --------------------------------------------------------------

@dataclass
class Camera:
    """An orbit camera. The target is what the scene is framed on."""

    target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    distance: float = 10.0
    orientation: np.ndarray = field(default_factory=quat_identity)
    fov: float = 22.0
    orthographic: bool = False
    scene_radius: float = 1.0
    # +1 drags the object with the pointer; -1 orbits around it. Exposed
    # because which one feels right is a matter of habit, not of correctness.
    rotation_sense: float = 1.0

    # Publication figures usually want a long lens: less perspective distortion
    # makes a polyhedron read as its real shape rather than a foreshortened one.
    MIN_FOV = 4.0
    MAX_FOV = 70.0

    # -- framing ----------------------------------------------------------
    def frame(self, center, radius: float, margin: float = 1.08) -> None:
        self.target = np.asarray(center, float).copy()
        self.scene_radius = max(float(radius), 1e-3)
        half = math.radians(self.fov) / 2.0
        self.distance = self.scene_radius / math.sin(half) * margin

    def reset_orientation(self) -> None:
        self.orientation = quat_identity()

    def view_along(self, axis) -> None:
        """Point the camera down a given world axis (a crystallographic one)."""
        axis = np.asarray(axis, float)
        n = np.linalg.norm(axis)
        if n < 1e-12:
            return
        axis = axis / n
        forward = np.array([0.0, 0.0, -1.0])
        # rotation taking the default forward onto -axis
        v = np.cross(forward, -axis)
        c = float(np.dot(forward, -axis))
        if np.linalg.norm(v) < 1e-9:
            self.orientation = quat_identity() if c > 0 else \
                quat_from_axis_angle([0, 1, 0], math.pi)
            return
        angle = math.acos(max(-1.0, min(1.0, c)))
        # the camera rotation is the inverse of the rotation applied to the world
        q = quat_from_axis_angle(v, angle)
        self.orientation = np.array([q[0], -q[1], -q[2], -q[3]])

    # -- interaction ------------------------------------------------------
    def drag_rotate(self, x0: float, y0: float, x1: float, y1: float,
                    width: int, height: int) -> None:
        if (x0, y0) == (x1, y1):
            return
        a = _arcball_vector(x0, y0, width, height)
        b = _arcball_vector(x1, y1, width, height)
        axis = np.cross(a, b)
        if np.linalg.norm(axis) < 1e-9:
            return
        angle = math.acos(max(-1.0, min(1.0, float(np.dot(a, b)))))
        # The sign is the "grab the object" convention: dragging right turns
        # the near face of the structure to the right, as though a hand were
        # on it. The opposite sign orbits a fixed object instead, which reads
        # as inverted to anyone used to a molecular viewer.
        delta = quat_from_axis_angle(axis, angle * 2.0 * self.rotation_sense)
        self.orientation = quat_multiply(delta, self.orientation)

    def drag_pan(self, dx: float, dy: float, width: int, height: int) -> None:
        """Pan so the point under the pointer keeps up with it."""
        if height <= 0:
            return
        scale = (self.half_height() * 2.0) / height
        r = quat_to_matrix(self.orientation)
        right, up = r[0, :3], r[1, :3]
        self.target = self.target - right * (dx * scale) + up * (dy * scale)

    def zoom(self, steps: float) -> None:
        self.distance = float(np.clip(self.distance * (0.88 ** steps),
                                      self.scene_radius * 0.05,
                                      self.scene_radius * 60.0))

    def set_fov(self, fov: float) -> None:
        """Change the lens while keeping the framing.

        Without the distance correction, changing the field of view zooms, which
        is not what a lens control should do.
        """
        old = math.sin(math.radians(self.fov) / 2.0)
        self.fov = float(np.clip(fov, self.MIN_FOV, self.MAX_FOV))
        new = math.sin(math.radians(self.fov) / 2.0)
        if old > 0 and new > 0:
            self.distance *= old / new

    # -- matrices ---------------------------------------------------------
    def eye(self) -> np.ndarray:
        r = quat_to_matrix(self.orientation)
        return self.target + r[2, :3] * self.distance

    def half_height(self) -> float:
        if self.orthographic:
            return self.distance * math.tan(math.radians(self.fov) / 2.0)
        return self.distance * math.tan(math.radians(self.fov) / 2.0)

    def view_matrix(self) -> np.ndarray:
        r = quat_to_matrix(self.orientation)
        m = np.eye(4)
        m[:3, :3] = r
        m[:3, 3] = -r @ self.target
        m[2, 3] -= self.distance
        return m

    def clip_planes(self) -> tuple[float, float]:
        span = self.scene_radius * 1.6
        near = max(self.distance - span, self.scene_radius * 1e-3, 1e-3)
        far = self.distance + span
        return near, far

    def projection_matrix(self, aspect: float) -> np.ndarray:
        near, far = self.clip_planes()
        if self.orthographic:
            return orthographic(self.half_height(), aspect, -far, far)
        return perspective(self.fov, aspect, near, far)

    def matrices(self, aspect: float) -> tuple[np.ndarray, np.ndarray]:
        return self.view_matrix(), self.projection_matrix(aspect)

    # -- projection helpers -----------------------------------------------
    def project(self, points, width: int, height: int) -> np.ndarray:
        """World points to screen pixels, plus view depth.

        Returns ``(n, 3)`` of ``(x_px, y_px, view_z)`` with y measured downwards
        from the top, matching Qt. ``view_z`` is negative in front of the camera.
        """
        p = np.atleast_2d(np.asarray(points, float))
        view, proj = self.matrices(width / max(height, 1))
        cam = (view @ np.hstack([p, np.ones((len(p), 1))]).T).T
        clip = (proj @ cam.T).T
        w = clip[:, 3].copy()
        w[np.abs(w) < 1e-12] = 1e-12
        ndc = clip[:, :3] / w[:, None]
        out = np.empty((len(p), 3))
        out[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * width
        out[:, 1] = (0.5 - ndc[:, 1] * 0.5) * height
        out[:, 2] = cam[:, 2]
        return out
