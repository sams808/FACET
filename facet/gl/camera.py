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
    # Where the drawn contents are, as distinct from where the camera pivots.
    # They coincide until someone chooses a rotation centre or pans, and every
    # depth calculation below has to use this one rather than `target`.
    scene_center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # How close the eye may come. Set from the largest drawn atom, so that
    # "centre on this atom and zoom right in" cannot put the eye inside it.
    min_distance: float = 0.0
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
        self.scene_center = np.asarray(center, float).copy()
        self.scene_radius = max(float(radius), 1e-3)
        half = math.radians(self.fov) / 2.0
        self.distance = self.scene_radius / math.sin(half) * margin

    def set_bounds(self, center, radius: float,
                   min_distance: float = 0.0) -> None:
        """Adopt what is drawn without moving the camera.

        Called on every rebuild, framing or not. Without it the camera keeps
        the extent of whatever was drawn when it last framed -- so enlarging the
        cell range clips the new atoms away, which it does today.
        """
        self.scene_center = np.asarray(center, float).copy()
        self.scene_radius = max(float(radius), 1e-3)
        self.min_distance = max(float(min_distance), 0.0)

    def center_on(self, point) -> None:
        """Put ``point`` at the centre of the view, and pivot there.

        Under perspective the eye is moved along its own axis by the depth the
        new centre gains or loses, so this is a pure pan: every drawn point
        keeps the depth it had and nothing grows or shrinks. Under an
        orthographic projection ``distance`` is not a depth at all, it is the
        zoom -- ``half_height`` is ``distance * tan(fov/2)``, and the QPainter
        tier reads it for its own scale -- so it is left alone, and the pan is
        exact without it.
        """
        point = np.asarray(point, float).copy()
        if not self.orthographic:
            forward = -quat_to_matrix(self.orientation)[2, :3]
            along = float((point - self.target) @ forward)
            self.distance = float(np.clip(
                self.distance + along,
                max(self.scene_radius * 0.05, self.min_distance),
                self.scene_radius * 60.0))
        self.target = point

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
                                      max(self.scene_radius * 0.05,
                                          self.min_distance),
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

    def scene_depth(self) -> float:
        """Axial depth of the scene centre from the eye.

        Equal to ``distance`` exactly while the pivot is the scene centre --
        which is what ``frame`` arranges -- so every window measured from it is
        unchanged at the default framing. It is the quantity the shipped code
        meant by ``distance`` in the two places below.
        """
        r = quat_to_matrix(self.orientation)
        offset = np.asarray(self.scene_center, float) - np.asarray(self.target,
                                                                   float)
        return float(self.distance - r[2, :3] @ offset)

    def clip_planes(self) -> tuple[float, float]:
        """Near and far, around the scene rather than around the pivot.

        Measured from `scene_depth`, the depth range of the drawn contents is
        exactly [d - r, d + r], so a span of 1.6 r always contains them however
        far the pivot has moved. Widening the span by the pivot offset instead
        would also contain them, and would spend depth-buffer precision -- 60%
        wider on a measured case -- that this does not need.
        """
        depth = self.scene_depth()
        span = self.scene_radius * 1.6
        near = max(depth - span, self.scene_radius * 1e-3, 1e-3)
        far = depth + span
        return near, far

    def fog_range(self) -> tuple[float, float]:
        """Where depth cueing starts and ends, in view depth.

        Read by the renderer instead of its own `distance +/- radius * k`, so
        that the cue stays on the structure when the pivot is off-centre. Same
        numbers as before while the two coincide.
        """
        depth = self.scene_depth()
        return depth - self.scene_radius * 0.55, depth + self.scene_radius * 1.25

    def projection_matrix(self, aspect: float) -> np.ndarray:
        near, far = self.clip_planes()
        if self.orthographic:
            return orthographic(self.half_height(), aspect, -far, far)
        return perspective(self.fov, aspect, near, far)

    def matrices(self, aspect: float) -> tuple[np.ndarray, np.ndarray]:
        return self.view_matrix(), self.projection_matrix(aspect)

    # -- stereo -----------------------------------------------------------
    # The separation is an angle, not a distance: half the angle the two eyes
    # subtend at the target. Expressing it that way makes it independent of how
    # far away the camera is, so the depth impression stays the same as you zoom
    # -- with a fixed distance it would collapse on a small cell and become
    # painful on a large one. About 1.2 degrees is the usual comfortable value
    # for a screen at arm's length.
    STEREO_SEPARATION = 1.2         # degrees, half-angle

    def for_eye(self, eye: int, separation: float | None = None) -> "Camera":
        """A copy of this camera displaced for one eye.

        ``eye`` is -1 for the left, +1 for the right, 0 for the original. A whole
        camera rather than just matrices, so every render path -- the GL tiers,
        the QPainter fallback, the image export -- gets stereo by being handed a
        different camera, with no stereo-specific code of its own.

        Toe-in stereo: each eye is rotated about the target, so both still look
        at the same point. It is not how human eyes work in detail -- it leaves a
        small vertical parallax at the edges of a wide field -- but it needs no
        off-axis projection, which is what keeps it identical on all three tiers,
        and at these angles the error is far below what the eye notices.
        """
        if not eye:
            return self
        angle = math.radians(separation if separation is not None
                             else self.STEREO_SEPARATION) * float(eye)
        up = quat_to_matrix(self.orientation)[1, :3]
        turn = quat_from_axis_angle(up, angle)
        # Every field, listed. A field left out here silently reverts to its
        # default in one eye only, which shows up as the two eyes clipping or
        # cueing differently -- the hardest kind of stereo fault to see.
        # tests/test_stereo_transparency.py walks dataclasses.fields against
        # this call so the next field added cannot be forgotten.
        return Camera(
            target=self.target.copy(), distance=self.distance,
            orientation=quat_multiply(self.orientation, turn),
            fov=self.fov, orthographic=self.orthographic,
            scene_radius=self.scene_radius,
            scene_center=np.asarray(self.scene_center, float).copy(),
            min_distance=self.min_distance,
            rotation_sense=self.rotation_sense)

    def eye_matrices(self, aspect: float, eye: int,
                     separation: float | None = None):
        """View and projection for one eye. ``eye`` is -1 left, +1 right, 0 mono."""
        return self.for_eye(eye, separation).matrices(aspect)

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
