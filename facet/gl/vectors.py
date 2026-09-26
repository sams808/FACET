"""The bond-valence vector sum, drawn.

What people ask for is a picture of the lone pair. What can be measured from a
structure is the **bond-valence vector sum**

    V = sum_i v_i u_i

over the contacts above the bond threshold, with ``u_i`` the unit vector from
the cation to ligand *i*. Its direction, negated, is the direction in which the
cation is *not* bonded, and for a cation with an ns^2 configuration that is the
direction an ns^2 lone pair is conventionally described as occupying. So the
lobe drawn here is the vector sum, and the module and the interface both say so.

Three properties make this honest to draw.

*The direction is invariant to an error in R0.* Every valence carries the same
factor exp(delta/b) when R0 moves by delta, and a common positive factor cannot
rotate a sum. The same cancellation makes ``phi = |V| / sum(v_i)`` exactly
R0-invariant, which is why phi and not |V| sets the drawn length -- |V| in
valence units is *not* invariant and belongs in the readout, not in the
geometry.

*It self-suppresses.* phi is zero at a centrosymmetric site, so nothing is
drawn there rather than an arbitrary direction being chosen out of numerical
noise.

*Only the scale is imported.* Length = scale x phi x mean bond length. Every
factor but the scale is measured from this structure. Placing the tip at a
literature lone-pair distance would put a number FACET cannot measure into
something that looks like a measurement, so that is not done.

The void cone is the other half of the same picture and is drawn separately: a
real cone of the measured void half-angle about the void axis. Where several
equally wide cones exist the returned axis is one of them, which the panel
states.

Nothing here imports Qt. Geometry is geometry; the renderers consume it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# The lobe's waist, in angstrom. Deliberately thinner than any drawn atom so a
# lobe cannot be mistaken for one, and so several lobes in a dense structure
# stay separable.
LOBE_RADIUS = 0.22

# Tessellation. 16 x 12 is 384 triangles per lobe, which is fewer than one
# drawn sphere on the FULL tier and still smooth at a metre from the screen.
AROUND = 16
ALONG = 12


@dataclass
class OverlayMesh:
    """A triangle soup made of independent parts, drawn as one call.

    ``anchors`` and ``part_of`` keep the parts addressable after concatenation:
    the slab filter keeps or drops a whole lobe by the position of the atom it
    belongs to, never by a per-triangle centroid, which would slice one in half.

    ``part_of`` is one index per *triangle*, not a start-and-end pair per part.
    That is deliberate: the renderer sorts transparent triangles back to front,
    which permutes them out of part order, and index ranges would quietly stop
    addressing anything the moment it did.
    """

    vertices: np.ndarray
    normals: np.ndarray
    color: tuple[float, float, float]
    alpha: float
    anchors: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), np.float32))
    part_of: np.ndarray = field(
        default_factory=lambda: np.zeros(0, np.int32))
    atoms: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    label: str = ""

    @property
    def n_triangles(self) -> int:
        return len(self.vertices) // 3

    @property
    def n_parts(self) -> int:
        return len(self.anchors)

    def translated(self, shift) -> "OverlayMesh":
        shift = np.asarray(shift, np.float32)
        return OverlayMesh(
            vertices=(self.vertices + shift).astype(np.float32),
            normals=self.normals.copy(), color=self.color, alpha=self.alpha,
            anchors=(self.anchors + shift).astype(np.float32),
            part_of=self.part_of.copy(), atoms=self.atoms.copy(),
            label=self.label)

    def reorder(self, triangle_index) -> None:
        """Permute the triangles in place, carrying the part mapping along."""
        index = np.asarray(triangle_index, int)
        triangles = np.asarray(self.vertices, np.float32).reshape(-1, 3, 3)
        normals = np.asarray(self.normals, np.float32).reshape(-1, 3, 3)
        self.vertices = triangles[index].reshape(-1, 3)
        self.normals = normals[index].reshape(-1, 3)
        if len(self.part_of) == len(index):
            self.part_of = np.asarray(self.part_of, np.int32)[index]

    def keep_parts(self, keep) -> "OverlayMesh":
        """A copy holding only the parts ``keep`` selects."""
        keep = np.asarray(keep, bool)
        if not self.n_parts or keep.all():
            return self
        if len(self.part_of) != self.n_triangles:
            return self
        alive = keep[np.asarray(self.part_of, int)]
        triangles = np.asarray(self.vertices, np.float32).reshape(-1, 3, 3)
        normals = np.asarray(self.normals, np.float32).reshape(-1, 3, 3)
        # renumber the surviving parts so part_of still indexes anchors
        renumber = np.cumsum(keep) - 1
        return OverlayMesh(
            vertices=triangles[alive].reshape(-1, 3),
            normals=normals[alive].reshape(-1, 3),
            color=self.color, alpha=self.alpha,
            anchors=self.anchors[keep].astype(np.float32),
            part_of=renumber[np.asarray(self.part_of, int)[alive]].astype(
                np.int32),
            atoms=self.atoms[keep] if len(self.atoms) else self.atoms,
            label=self.label)


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------

@dataclass
class VectorSums:
    """Per drawn atom: the vector sum, the scalar sum, phi, and the count."""

    vector: np.ndarray        # (n, 3) in valence units, double precision
    total: np.ndarray         # (n,)   sum of the valences
    phi: np.ndarray           # (n,)   |vector| / total, 0 where total is 0
    count: np.ndarray         # (n,)   contacts above the threshold
    mean_distance: np.ndarray  # (n,)  mean bonded distance, for the scale

    def __len__(self) -> int:
        return len(self.total)


def accumulate(bond_a, bond_b, bond_atoms, bond_cation, bond_valence,
               v_bond: float, n_atoms: int, occupancy=None) -> VectorSums:
    """Sum the valence vectors on every drawn atom.

    Works from the scene's cached bond arrays rather than from a fresh
    neighbour search, for two reasons: it is exact for every drawn atom
    including the periodic images added to close a bond, and it is cheap enough
    to redo while the threshold slider is moving.

    ``occupancy`` weights each contact by its ligand's occupancy, which is what
    ``coordination.analyse_structure`` does when it forms the same sum through
    ``bv.phi_index``. Leaving it out made the drawn phi differ from the reported
    one by 0.13 on a partially occupied site.

    ``bond_cation[k]`` says which end of bond *k* is the cation: 0 for
    ``bond_atoms[k, 0]``, 1 for ``bond_atoms[k, 1]``, and -1 for a contact
    with no cation (which is not summed). The cation's own direction to the
    ligand is taken from the drawn endpoints, so an image cation contributes
    the vector of the bond as drawn rather than of some other copy -- a cell
    translation cannot rotate the sum, which is exactly why this is safe and
    why propagating a representative site's vectors by translation was not.
    """
    vector = np.zeros((n_atoms, 3), np.float64)
    total = np.zeros(n_atoms, np.float64)
    count = np.zeros(n_atoms, np.int32)
    length = np.zeros(n_atoms, np.float64)

    bond_valence = np.asarray(bond_valence, np.float64)
    if occupancy is not None and len(occupancy) == len(bond_valence):
        weight = bond_valence * np.asarray(occupancy, np.float64)
    else:
        weight = bond_valence
    if not len(bond_valence) or n_atoms == 0:
        return VectorSums(vector, total, np.zeros(n_atoms), count, length)

    bond_atoms = np.asarray(bond_atoms, np.int64).reshape(-1, 2)
    bond_cation = np.asarray(bond_cation, np.int64).reshape(-1)
    delta = (np.asarray(bond_b, np.float64)
             - np.asarray(bond_a, np.float64))
    norm = np.linalg.norm(delta, axis=1)

    keep = ((bond_valence >= float(v_bond)) & (bond_cation >= 0)
            & (norm > 1e-9))
    if not keep.any():
        return VectorSums(vector, total, np.zeros(n_atoms), count, length)

    unit = delta[keep] / norm[keep][:, None]
    which = bond_cation[keep]
    index = np.where(which == 0, bond_atoms[keep, 0], bond_atoms[keep, 1])
    # cation -> anion: the bond as drawn when the cation is the first atom, the
    # reverse when it is the second
    sign = np.where(which == 0, 1.0, -1.0)
    v = weight[keep]

    inside = (index >= 0) & (index < n_atoms)
    index, unit, sign, v = index[inside], unit[inside], sign[inside], v[inside]
    np.add.at(vector, index, (v * sign)[:, None] * unit)
    np.add.at(total, index, v)
    np.add.at(count, index, 1)
    np.add.at(length, index, norm[keep][inside])

    with np.errstate(invalid="ignore", divide="ignore"):
        phi = np.where(total > 0, np.linalg.norm(vector, axis=1) / total, 0.0)
        mean = np.where(count > 0, length / np.maximum(count, 1), 0.0)
    return VectorSums(vector, total, phi, count, mean)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def _frames(directions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A right-handed frame (e1, e2, d) about each unit direction."""
    d = np.asarray(directions, np.float64)
    d = d / np.maximum(np.linalg.norm(d, axis=1)[:, None], 1e-12)
    # any axis not parallel to d; choosing by the smallest component keeps the
    # cross product well conditioned
    helper = np.zeros_like(d)
    helper[np.arange(len(d)), np.argmin(np.abs(d), axis=1)] = 1.0
    e1 = np.cross(helper, d)
    e1 /= np.maximum(np.linalg.norm(e1, axis=1)[:, None], 1e-12)
    e2 = np.cross(d, e1)
    return e1, e2, d


def _revolve(origins, directions, radii, heights, dradii, dheights,
             around: int) -> tuple[np.ndarray, np.ndarray]:
    """Turn per-atom profiles into surfaces of revolution.

    ``radii``/``heights`` are (n, m) profiles sampled along each axis, and the
    derivatives give the analytic normal, which stays correct at a tip where a
    face-averaged normal degenerates.
    """
    n, m = radii.shape
    e1, e2, d = _frames(directions)
    theta = np.linspace(0.0, 2.0 * np.pi, around, endpoint=False)
    cos, sin = np.cos(theta), np.sin(theta)

    # radial unit vector per (atom, angle) -> (n, around, 3)
    e_r = cos[None, :, None] * e1[:, None, :] + sin[None, :, None] * e2[:, None, :]

    points = (radii[:, :, None, None] * e_r[:, None, :, :]
              + heights[:, :, None, None] * d[:, None, None, :])
    points = points + np.asarray(origins, np.float64)[:, None, None, :]

    normals = (dheights[:, :, None, None] * e_r[:, None, :, :]
               - dradii[:, :, None, None] * d[:, None, None, :])
    normals /= np.maximum(np.linalg.norm(normals, axis=3)[..., None], 1e-12)
    return points, normals            # (n, m, around, 3) each


def _quads_to_triangles(points, normals):
    """Two triangles per quad of the (m, around) grid, as a soup."""
    n, m, around, _ = points.shape
    nxt = (np.arange(around) + 1) % around
    p00 = points[:, :-1, :, :]
    p01 = points[:, :-1, :, :][:, :, nxt, :]
    p10 = points[:, 1:, :, :]
    p11 = points[:, 1:, :, :][:, :, nxt, :]
    n00 = normals[:, :-1, :, :]
    n01 = normals[:, :-1, :, :][:, :, nxt, :]
    n10 = normals[:, 1:, :, :]
    n11 = normals[:, 1:, :, :][:, :, nxt, :]

    tri_p = np.stack([p00, p10, p11, p00, p11, p01], axis=3)
    tri_n = np.stack([n00, n10, n11, n00, n11, n01], axis=3)
    per_atom = (m - 1) * around * 6
    return (tri_p.reshape(n, per_atom, 3).astype(np.float32),
            tri_n.reshape(n, per_atom, 3).astype(np.float32))


def lobe_meshes(origins, directions, lengths, *, radius: float = LOBE_RADIUS,
                around: int = AROUND, along: int = ALONG, atoms=None,
                color=(0.86, 0.72, 0.98), alpha: float = 0.55,
                label: str = "bond-valence vector") -> OverlayMesh:
    """A teardrop lobe per entry, anchored at ``origins``.

    The profile is ``r(t) = radius * sin(pi t) ** 0.75`` against ``z = L t``:
    pointed at the atom, widest a little past the middle, pointed at the far
    end. Not a cone -- a cone implies an aperture and the vector sum has none --
    and not a translucent sphere, which reads as another atom.
    """
    origins = np.atleast_2d(np.asarray(origins, np.float64))
    directions = np.atleast_2d(np.asarray(directions, np.float64))
    lengths = np.atleast_1d(np.asarray(lengths, np.float64))
    if not len(origins):
        return OverlayMesh(np.zeros((0, 3), np.float32),
                           np.zeros((0, 3), np.float32), color, alpha,
                           label=label)

    t = np.linspace(0.0, 1.0, along + 1)
    profile = np.sin(np.pi * t) ** 0.75
    # d/dt of the profile; infinite at the tips, which is what makes the
    # analytic normal turn to face along the axis there
    with np.errstate(invalid="ignore", divide="ignore"):
        dprofile = np.gradient(profile, t)
    dprofile = np.clip(np.nan_to_num(dprofile), -40.0, 40.0)

    radii = radius * profile[None, :] * np.ones((len(origins), 1))
    heights = lengths[:, None] * t[None, :]
    dradii = radius * dprofile[None, :] * np.ones((len(origins), 1))
    dheights = lengths[:, None] * np.ones((1, len(t)))

    points, normals = _revolve(origins, directions, radii, heights,
                               dradii, dheights, around)
    tri_p, tri_n = _quads_to_triangles(points, normals)
    per_part = tri_p.shape[1] // 3                     # triangles, not vertices
    part_of = np.repeat(np.arange(len(origins), dtype=np.int32), per_part)
    return OverlayMesh(
        vertices=tri_p.reshape(-1, 3), normals=tri_n.reshape(-1, 3),
        color=tuple(float(c) for c in color), alpha=float(alpha),
        anchors=origins.astype(np.float32), part_of=part_of,
        atoms=(np.asarray(atoms, np.int32) if atoms is not None
               else np.arange(len(origins), dtype=np.int32)),
        label=label)


def cone_meshes(origins, axes, half_angles, lengths, *, around: int = AROUND,
                atoms=None, color=(0.40, 0.78, 0.92), alpha: float = 0.28,
                label: str = "void cone") -> OverlayMesh:
    """An open cone per entry: apex at the atom, the measured half-angle.

    ``half_angles`` in degrees, as the analysis reports them. The lateral
    surface only: a cone closed with a disc reads as a solid, and what is being
    shown is an angle.
    """
    origins = np.atleast_2d(np.asarray(origins, np.float64))
    axes = np.atleast_2d(np.asarray(axes, np.float64))
    half = np.atleast_1d(np.asarray(half_angles, np.float64))
    lengths = np.atleast_1d(np.asarray(lengths, np.float64))
    if not len(origins):
        return OverlayMesh(np.zeros((0, 3), np.float32),
                           np.zeros((0, 3), np.float32), color, alpha,
                           label=label)

    alpha_rad = np.clip(np.radians(half), 1e-3, np.pi / 2 - 1e-3)
    # two rings: the apex and the rim, so the lateral surface is one quad band
    t = np.array([0.0, 1.0])
    radii = (lengths * np.tan(alpha_rad))[:, None] * t[None, :]
    heights = lengths[:, None] * t[None, :]
    # a straight generator: dr/dt and dz/dt are constant along it
    dradii = np.repeat((lengths * np.tan(alpha_rad))[:, None], 2, axis=1)
    dheights = np.repeat(lengths[:, None], 2, axis=1)

    points, normals = _revolve(origins, axes, radii, heights,
                               dradii, dheights, around)
    tri_p, tri_n = _quads_to_triangles(points, normals)
    per_part = tri_p.shape[1] // 3                     # triangles, not vertices
    part_of = np.repeat(np.arange(len(origins), dtype=np.int32), per_part)
    return OverlayMesh(
        vertices=tri_p.reshape(-1, 3), normals=tri_n.reshape(-1, 3),
        color=tuple(float(c) for c in color), alpha=float(alpha),
        anchors=origins.astype(np.float32), part_of=part_of,
        atoms=(np.asarray(atoms, np.int32) if atoms is not None
               else np.arange(len(origins), dtype=np.int32)),
        label=label)
