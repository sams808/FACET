"""Lattice planes and slabs.

Two related things that VESTA and CrystalMaker both offer:

* a **lattice plane** -- the (hkl) planes drawn as translucent sheets, so you
  can see how a layer sits relative to the polyhedra. Drawing the plane means
  finding where it cuts the drawn box, which is a convex polygon.
* a **slab** -- showing only what lies between two parallel (hkl) planes. This
  is how you look at one layer of a layered structure without the layers above
  and below in the way, and it is the honest version of "hide the atoms I do not
  want": the criterion is a stated distance along a stated normal, not a click.

The normal of the (hkl) plane is the reciprocal-lattice vector
``h a* + k b* + l c*``, never ``h a + k b + l c``. Those agree only in a cubic
cell. A monoclinic beta of 113 degrees puts the two more than twenty degrees
apart, so using the real-space combination silently draws the wrong planes for
most of the structures this program exists for.

Nothing here decides which plane is interesting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .structure import Structure

# A plane is "at" an atom if the atom is within this distance of it, in angstrom.
# Loose enough to catch a puckered layer, tight enough not to catch the next one.
ON_PLANE = 0.30


@dataclass
class LatticePlane:
    """One set of (hkl) planes, at a chosen spacing along their normal.

    ``offset`` is in units of the interplanar spacing d, so offset 0 passes
    through the cell origin, 1 through the next plane of the family, 0.5 halfway
    between. Expressing it that way rather than in angstrom means the control
    behaves the same whatever the indices.
    """

    h: int
    k: int
    l: int
    offset: float = 0.0                       # in units of d
    color: tuple[float, float, float] = (0.36, 0.72, 0.92)
    alpha: float = 0.30
    visible: bool = True
    show_edges: bool = True
    repeat: int = 1                           # how many planes of the family

    @property
    def hkl(self) -> str:
        return f"{self.h} {self.k} {self.l}"

    @property
    def is_valid(self) -> bool:
        return (self.h, self.k, self.l) != (0, 0, 0)

    def label(self, structure: Structure | None = None) -> str:
        if structure is None or not self.is_valid:
            return f"({self.hkl})"
        return f"({self.hkl})  d = {spacing(structure, self):.4f} A"


@dataclass
class Slab:
    """Keep only what lies between two parallel (hkl) planes.

    ``centre`` is in units of d from the origin, like a plane's offset;
    ``thickness`` is in angstrom, because that is what you think in when you ask
    for "one layer". Outside the slab, atoms are not drawn and neither are the
    bonds that reach them.
    """

    h: int = 0
    k: int = 0
    l: int = 1
    centre: float = 0.0                       # in units of d
    thickness: float = 4.0                    # angstrom
    enabled: bool = False

    @property
    def hkl(self) -> str:
        return f"{self.h} {self.k} {self.l}"

    @property
    def is_valid(self) -> bool:
        return (self.h, self.k, self.l) != (0, 0, 0) and self.thickness > 0


# ---------------------------------------------------------------------------
# the geometry of one plane
# ---------------------------------------------------------------------------

def normal_and_spacing(structure: Structure, h: int, k: int, l: int):
    """The unit normal of (hkl) and the interplanar spacing d, in cartesian.

    Returns ``(normal, d)``. The normal is ``h a* + k b* + l c*`` normalised,
    and ``d`` is the reciprocal of that vector's length before normalising --
    which is the same d the diffraction code uses, from the same definition.
    """
    orth = np.asarray(structure.cell.orth, float)
    a, b, c = orth[:, 0], orth[:, 1], orth[:, 2]
    volume = float(np.dot(a, np.cross(b, c)))
    if abs(volume) < 1e-12:
        raise ValueError(f"{structure.name}: degenerate cell")
    reciprocal = np.stack([np.cross(b, c), np.cross(c, a), np.cross(a, b)]) / volume
    vector = np.array([h, k, l], float) @ reciprocal
    length = float(np.linalg.norm(vector))
    if length < 1e-12:
        raise ValueError("(0 0 0) is not a plane")
    return vector / length, 1.0 / length


def spacing(structure: Structure, plane: LatticePlane) -> float:
    return normal_and_spacing(structure, plane.h, plane.k, plane.l)[1]


def in_plane_axes(structure: Structure, h: int, k: int, l: int):
    """Two orthonormal in-plane directions and the normal.

    The first in-plane axis is chosen to lie along a short lattice direction
    where one is available, so a section drawn on the plane has axes that mean
    something crystallographic instead of an arbitrary rotation.
    """
    normal, _ = normal_and_spacing(structure, h, k, l)
    orth = np.asarray(structure.cell.orth, float)

    best, best_length = None, 0.0
    for candidate in (orth[:, 0], orth[:, 1], orth[:, 2],
                      orth[:, 0] + orth[:, 1], orth[:, 1] + orth[:, 2],
                      orth[:, 0] + orth[:, 2]):
        projected = candidate - np.dot(candidate, normal) * normal
        length = float(np.linalg.norm(projected))
        # prefer the shortest lattice direction that actually lies in the plane
        if length > 1e-6 and (best is None or length < best_length):
            best, best_length = projected / length, length
    if best is None:                                   # pragma: no cover
        helper = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(helper, normal))) > 0.9:
            helper = np.array([1.0, 0.0, 0.0])
        best = np.cross(helper, normal)
        best /= np.linalg.norm(best)
    second = np.cross(normal, best)
    return best, second / np.linalg.norm(second), normal


def plane_point(structure: Structure, plane: LatticePlane,
                index: int = 0) -> np.ndarray:
    """A cartesian point on the ``index``-th plane of the family."""
    normal, d = normal_and_spacing(structure, plane.h, plane.k, plane.l)
    return normal * (plane.offset + index) * d


def polygon_in_box(normal, point, corners) -> np.ndarray:
    """Where a plane cuts a convex box: the cross-section polygon.

    The box is given by its eight corners. Every edge is tested for a crossing,
    the crossing points are collected, and they are then sorted by angle about
    their own centroid -- which is valid because the cross-section of a convex
    body is convex, so angular order round the centroid is the boundary order.

    Returns an (n, 3) array of cartesian vertices, empty if the plane misses.
    """
    corners = np.asarray(corners, float)
    normal = np.asarray(normal, float)
    point = np.asarray(point, float)

    # the twelve edges of a box indexed as bits (x, y, z)
    edges = [(i, i ^ bit) for i in range(8) for bit in (1, 2, 4) if not i & bit]

    distances = (corners - point) @ normal
    points = []
    for i, j in edges:
        di, dj = distances[i], distances[j]
        if di == 0.0:
            points.append(corners[i])
        if dj == 0.0:
            points.append(corners[j])
        if di * dj < 0.0:
            t = di / (di - dj)
            points.append(corners[i] + t * (corners[j] - corners[i]))
    if len(points) < 3:
        return np.zeros((0, 3))

    points = np.asarray(points, float)
    # drop duplicates, which a plane through a corner produces
    keep = []
    for candidate in points:
        if not any(np.linalg.norm(candidate - other) < 1e-9 for other in keep):
            keep.append(candidate)
    if len(keep) < 3:
        return np.zeros((0, 3))
    points = np.asarray(keep, float)

    centroid = points.mean(axis=0)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(helper, normal))) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    u = np.cross(helper, normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    offsets = points - centroid
    angles = np.arctan2(offsets @ v, offsets @ u)
    return points[np.argsort(angles)]


def box_corners(structure: Structure, replicate=(1, 1, 1),
                margin: float = 0.0) -> np.ndarray:
    """The eight corners of the drawn box, in cartesian.

    ``margin`` grows the box outwards in angstrom, so a plane drawn across a
    structure extends slightly past the atoms instead of stopping exactly at
    them, which reads better and matches what VESTA does.
    """
    orth = np.asarray(structure.cell.orth, float)
    na, nb, nc = (max(int(n), 1) for n in replicate)

    # The margin extends the box along each lattice direction, which keeps it a
    # parallelepiped. Pushing the corners outwards radially instead would grow
    # the four diagonals by different fractions, the faces would stop being
    # planar, and a cross-section of the result would no longer be guaranteed
    # convex -- which is the one thing polygon_in_box relies on.
    grow = np.zeros(3)
    if margin > 0:
        for axis in range(3):
            length = float(np.linalg.norm(orth[:, axis]))
            if length > 1e-12:
                grow[axis] = margin / length

    low = -grow
    high = np.array([na, nb, nc], float) + grow

    ordered = np.zeros((8, 3))
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                fractional = np.array([high[0] if i else low[0],
                                       high[1] if j else low[1],
                                       high[2] if k else low[2]])
                # index = x | y<<1 | z<<2, the convention polygon_in_box uses
                ordered[i | (j << 1) | (k << 2)] = orth @ fractional
    return ordered


def plane_polygons(structure: Structure, plane: LatticePlane,
                   replicate=(1, 1, 1), margin: float = 0.4):
    """Cross-section polygons for every plane of the family that cuts the box.

    Returns a list of (n, 3) vertex arrays. ``plane.repeat`` planes are placed at
    successive multiples of d; those that miss the box entirely are left out
    rather than returned empty, so the caller can just count them.
    """
    if not plane.is_valid:
        return []
    normal, d = normal_and_spacing(structure, plane.h, plane.k, plane.l)
    corners = box_corners(structure, replicate, margin)
    polygons = []
    for index in range(max(int(plane.repeat), 1)):
        point = normal * (plane.offset + index) * d
        polygon = polygon_in_box(normal, point, corners)
        if len(polygon) >= 3:
            polygons.append(polygon)
    return polygons


def triangulate(polygon) -> tuple[np.ndarray, np.ndarray]:
    """A convex polygon as a triangle fan. Returns ``(vertices, normals)``.

    A fan is enough because a plane's cross-section through a convex box is
    always convex.
    """
    polygon = np.asarray(polygon, float)
    if len(polygon) < 3:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    normal = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
    length = float(np.linalg.norm(normal))
    normal = normal / length if length > 1e-12 else np.array([0.0, 0.0, 1.0])

    vertices = []
    for i in range(1, len(polygon) - 1):
        vertices.extend([polygon[0], polygon[i], polygon[i + 1]])
    vertices = np.asarray(vertices, np.float32)
    return vertices, np.tile(normal.astype(np.float32), (len(vertices), 1))


# ---------------------------------------------------------------------------
# slabs
# ---------------------------------------------------------------------------

def signed_distances(structure: Structure, points, h: int, k: int, l: int,
                     offset: float = 0.0) -> np.ndarray:
    """Distance of each cartesian point from the (hkl) plane, in angstrom.

    Signed along the plane normal, so the sign says which side.
    """
    normal, d = normal_and_spacing(structure, h, k, l)
    origin = normal * offset * d
    return (np.asarray(points, float).reshape(-1, 3) - origin) @ normal


def slab_mask(structure: Structure, points, slab: Slab) -> np.ndarray:
    """Which points lie inside the slab. All of them if it is off or invalid."""
    points = np.asarray(points, float).reshape(-1, 3)
    if not slab.enabled or not slab.is_valid:
        return np.ones(len(points), bool)
    distance = signed_distances(structure, points, slab.h, slab.k, slab.l,
                               slab.centre)
    return np.abs(distance) <= slab.thickness / 2.0


def atoms_in_slab(structure: Structure, slab: Slab) -> np.ndarray:
    """Indices of the structure's atoms that fall inside the slab."""
    if not structure.atoms:
        return np.zeros(0, int)
    cart = np.array([a.cart for a in structure.atoms], float)
    return np.nonzero(slab_mask(structure, cart, slab))[0]


def atoms_on_plane(structure: Structure, plane: LatticePlane,
                   tolerance: float = ON_PLANE) -> list[int]:
    """Atoms lying within ``tolerance`` of any plane of the family.

    Useful for answering "which atoms make up this layer", which is a question
    about the structure and not about the picture. The answer is a list of atom
    indices and nothing else -- no statement about whether they form a layer.
    """
    if not plane.is_valid or not structure.atoms:
        return []
    normal, d = normal_and_spacing(structure, plane.h, plane.k, plane.l)
    cart = np.array([a.cart for a in structure.atoms], float)
    along = cart @ normal
    # distance to the nearest plane of the family, allowing for the offset
    phase = (along / d) - plane.offset
    nearest = np.abs(phase - np.round(phase)) * d
    return [int(i) for i in np.nonzero(nearest <= tolerance)[0]]


def plane_occupancy(structure: Structure, plane: LatticePlane,
                    tolerance: float = ON_PLANE) -> dict[str, float]:
    """What sits on the planes of this family: element counts, occupancy-weighted.

    A measurement, reported as it is found. Whether a family of planes that
    happens to contain only one element constitutes a layer is not decided here.
    """
    out: dict[str, float] = {}
    for index in atoms_on_plane(structure, plane, tolerance):
        atom = structure.atoms[index]
        out[atom.element] = out.get(atom.element, 0.0) + atom.occupancy
    return out


def interplanar_angle(structure: Structure, first, second) -> float:
    """Angle between two (hkl) plane normals, in degrees.

    Computed on the reciprocal-lattice vectors, so it is right in any cell. The
    naive version -- the angle between ``h a + k b + l c`` vectors -- is a
    standard way to publish a wrong dihedral for a monoclinic structure.
    """
    first_normal, _ = normal_and_spacing(structure, *first)
    second_normal, _ = normal_and_spacing(structure, *second)
    if float(np.dot(first_normal, second_normal)) < 0:
        second_normal = -second_normal
    # atan2 of the cross and the dot, not acos of the dot. acos loses half its
    # digits where the cosine approaches 1, which is exactly the case that
    # matters: two nearly parallel planes, and a plane with itself, came out as
    # 1e-3 degrees instead of zero.
    cross = float(np.linalg.norm(np.cross(first_normal, second_normal)))
    dot = float(np.dot(first_normal, second_normal))
    return math.degrees(math.atan2(cross, dot))
