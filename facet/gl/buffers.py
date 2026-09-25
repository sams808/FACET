"""Turning a :class:`~facet.gl.scene.Scene` into flat vertex arrays.

Pure numpy, so the geometry can be checked without a graphics context -- which
matters, because a transposed axis in a tube basis looks like a rendering bug
and is very hard to find by eye.

Everything is expanded on the CPU rather than instanced: the compatible tier has
no instancing at all, and one geometry path means the two tiers cannot drift
apart. At FACET's scale the extra vertex data is negligible -- a 5000-atom cell
is 20000 sphere vertices, under a megabyte.
"""
from __future__ import annotations

import numpy as np

from .scene import Scene

# The four corners of a billboard, as two triangles.
_QUAD = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0],
                  [-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], np.float32)
VERTS_PER_SPHERE = 6


# --- picking -----------------------------------------------------------------
# An object id is written into an 8-bit RGB attachment and read back one pixel
# at a time. Exact by construction: no ray has to be intersected on the CPU, so
# there is no near-miss case and no disagreement between what is drawn and what
# is picked. Zero is reserved for "nothing", so ids are stored offset by one.

def id_to_rgb(index) -> np.ndarray:
    """Encode object indices as normalised RGB triples."""
    i = np.asarray(index, np.int64) + 1
    r = (i & 0xFF).astype(np.float32) / 255.0
    g = ((i >> 8) & 0xFF).astype(np.float32) / 255.0
    b = ((i >> 16) & 0xFF).astype(np.float32) / 255.0
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def rgb_to_id(r: int, g: int, b: int) -> int | None:
    """Decode a pixel back to an object index, or None for the background."""
    packed = int(r) | (int(g) << 8) | (int(b) << 16)
    return packed - 1 if packed else None


# --- spheres -----------------------------------------------------------------

def sphere_vertices(scene: Scene) -> dict[str, np.ndarray]:
    """Billboard quads for every atom.

    Returns one array per shader attribute, each with
    ``6 * n_atoms`` rows.
    """
    n = scene.n_atoms
    if n == 0:
        return _empty_sphere()

    corner = np.tile(_QUAD, (n, 1))
    repeat = VERTS_PER_SPHERE
    return {
        "aCorner": corner,
        "aCenter": np.repeat(scene.atom_position, repeat, axis=0),
        "aRadius": np.repeat(scene.atom_radius, repeat).reshape(-1, 1),
        "aColor": np.repeat(scene.atom_color, repeat, axis=0),
        "aId": np.repeat(id_to_rgb(scene.atom_index), repeat, axis=0),
    }


def _empty_sphere() -> dict[str, np.ndarray]:
    return {"aCorner": np.zeros((0, 2), np.float32),
            "aCenter": np.zeros((0, 3), np.float32),
            "aRadius": np.zeros((0, 1), np.float32),
            "aColor": np.zeros((0, 3), np.float32),
            "aId": np.zeros((0, 3), np.float32)}


# --- bonds -------------------------------------------------------------------

def unit_tube(sides: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A unit tube: radius 1, running 0..1 along +z, split at the midpoint.

    Split at the midpoint so that the colour test in the shader lands on a real
    vertex boundary. Interpolating a colour across the seam instead would give
    a smeared gradient where a crisp division is wanted.
    """
    pos, nrm, param = [], [], []
    for half, (z0, z1) in enumerate(((0.0, 0.5), (0.5, 1.0))):
        for i in range(sides):
            a0 = 2.0 * np.pi * i / sides
            a1 = 2.0 * np.pi * (i + 1) / sides
            p0 = (np.cos(a0), np.sin(a0))
            p1 = (np.cos(a1), np.sin(a1))
            for xy, z in ((p0, z0), (p1, z0), (p1, z1),
                          (p0, z0), (p1, z1), (p0, z1)):
                pos.append((xy[0], xy[1], z))
                nrm.append((xy[0], xy[1], 0.0))
                param.append(0.0 if half == 0 else 1.0)
    return (np.array(pos, np.float32), np.array(nrm, np.float32),
            np.array(param, np.float32))


def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two vectors perpendicular to each row of `axis`.

    The helper vector is chosen per row rather than globally: a single fixed
    helper becomes parallel to the axis for some bond in almost any structure,
    and the cross product then collapses to zero.
    """
    helper = np.where(np.abs(axis[:, 2:3]) < 0.9,
                      np.array([0.0, 0.0, 1.0], np.float32),
                      np.array([1.0, 0.0, 0.0], np.float32))
    x = np.cross(helper, axis)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    x = x / n
    y = np.cross(axis, x)
    return x.astype(np.float32), y.astype(np.float32)


def tube_vertices(scene: Scene, sides: int = 16) -> dict[str, np.ndarray]:
    """Expand every bond into a tube mesh in world space."""
    m = scene.n_bonds
    if m == 0:
        return _empty_tube()

    tpos, tnrm, tparam = unit_tube(sides)
    k = len(tpos)

    axis = scene.bond_b - scene.bond_a
    length = np.linalg.norm(axis, axis=1, keepdims=True)
    length[length < 1e-9] = 1e-9
    axis = (axis / length).astype(np.float32)
    bx, by = _orthonormal_basis(axis)

    # basis[i] maps unit-tube coordinates into world space for bond i
    radius = scene.bond_radius.reshape(-1, 1)
    local = np.tile(tpos, (m, 1))                      # (m*k, 3)
    bx_r = np.repeat(bx, k, axis=0)
    by_r = np.repeat(by, k, axis=0)
    ax_r = np.repeat(axis, k, axis=0)
    rad_r = np.repeat(radius, k, axis=0)
    len_r = np.repeat(length, k, axis=0)
    origin = np.repeat(scene.bond_a, k, axis=0)

    position = (origin
                + bx_r * (local[:, 0:1] * rad_r)
                + by_r * (local[:, 1:2] * rad_r)
                + ax_r * (local[:, 2:3] * len_r))

    local_n = np.tile(tnrm, (m, 1))
    normal = (bx_r * local_n[:, 0:1] + by_r * local_n[:, 1:2])
    nn = np.linalg.norm(normal, axis=1, keepdims=True)
    nn[nn < 1e-12] = 1.0
    normal = normal / nn

    return {
        "aPosition": position.astype(np.float32),
        "aNormal": normal.astype(np.float32),
        "aColor": np.repeat(scene.bond_color_a, k, axis=0),
        "aColorB": np.repeat(scene.bond_color_b, k, axis=0),
        "aParam": np.tile(tparam, m).reshape(-1, 1),
    }


def _empty_tube() -> dict[str, np.ndarray]:
    return {"aPosition": np.zeros((0, 3), np.float32),
            "aNormal": np.zeros((0, 3), np.float32),
            "aColor": np.zeros((0, 3), np.float32),
            "aColorB": np.zeros((0, 3), np.float32),
            "aParam": np.zeros((0, 1), np.float32)}


# --- lines -------------------------------------------------------------------

def line_vertices(segments: np.ndarray) -> np.ndarray:
    """Flatten ``(n, 2, 3)`` segment endpoints into a GL_LINES array."""
    if segments is None or len(segments) == 0:
        return np.zeros((0, 3), np.float32)
    return np.asarray(segments, np.float32).reshape(-1, 3)


# --- sizing ------------------------------------------------------------------

def estimate_vertex_count(scene: Scene, sides: int = 16) -> int:
    """How many vertices this scene will upload. Used to pick a tube resolution
    that keeps the software tier usable."""
    return (scene.n_atoms * VERTS_PER_SPHERE
            + scene.n_bonds * sides * 12
            + len(scene.poly_vertices)
            + len(scene.cell_segments) * 2)


def tube_sides_for(scene: Scene, budget: int = 400_000, maximum: int = 20,
                   minimum: int = 5) -> int:
    """Choose a tube resolution that fits a vertex budget.

    A large structure on the software tier is the case that matters: dropping
    from 16 sides to 6 is barely visible on a thin bond and is the difference
    between interactive and not.
    """
    if scene.n_bonds == 0:
        return maximum
    spare = budget - scene.n_atoms * VERTS_PER_SPHERE - len(scene.poly_vertices)
    if spare <= 0:
        return minimum
    sides = int(spare // max(scene.n_bonds * 12, 1))
    return int(np.clip(sides, minimum, maximum))
