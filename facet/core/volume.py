"""Volumetric data: grids, isosurfaces and sections.

A :class:`Grid` is a scalar field sampled on a regular mesh in fractional
coordinates, so it tiles with the lattice whatever the cell shape. Charge
densities read from CHGCAR, CUBE or XSF arrive as one; so does a bond-valence
map computed by FACET itself.

**Isosurfaces use marching tetrahedra, not marching cubes.** Each cube is split
into six tetrahedra, and a tetrahedron has only sixteen cases against two
hundred and fifty-six. More importantly it has no ambiguous ones: marching
cubes has saddle configurations where two readings are equally valid and a
wrong choice punches a hole in the surface. Marching tetrahedra produces more
triangles for the same field and never produces a hole.

It is also small enough to implement here. scikit-image has a marching-cubes
routine, but pulling in scikit-image to get one function would add tens of
megabytes to a distribution that has to be emailed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import bv, elements
from .structure import Structure


@dataclass
class Grid:
    """A scalar field on a regular mesh over one unit cell.

    `values` is indexed ``[i, j, k]`` with i along a, j along b, k along c, and
    is treated as periodic: index ``n`` is the same point as index ``0``.
    """

    values: np.ndarray
    cell: object                          # facet.core.structure.Cell
    name: str = "field"
    units: str = ""
    source: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.values.shape)

    @property
    def n_points(self) -> int:
        return int(self.values.size)

    @property
    def spacing(self) -> tuple[float, float, float]:
        """Sample spacing along each axis, in angstrom."""
        a, b, c = self.cell.lengths
        na, nb, nc = self.shape
        return (a / na, b / nb, c / nc)

    def statistics(self) -> dict:
        v = self.values
        return {"min": float(v.min()), "max": float(v.max()),
                "mean": float(v.mean()), "std": float(v.std()),
                "points": self.n_points}

    # -- sampling ----------------------------------------------------------
    def at_fractional(self, frac) -> np.ndarray:
        """Trilinear interpolation at fractional coordinates, periodic."""
        f = np.atleast_2d(np.asarray(frac, float)) % 1.0
        n = np.array(self.shape, float)
        g = f * n
        i0 = np.floor(g).astype(int)
        t = g - i0
        i0 = i0 % np.array(self.shape, int)
        i1 = (i0 + 1) % np.array(self.shape, int)

        out = np.zeros(len(f))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    weight = ((t[:, 0] if dx else 1 - t[:, 0])
                              * (t[:, 1] if dy else 1 - t[:, 1])
                              * (t[:, 2] if dz else 1 - t[:, 2]))
                    xi = i1[:, 0] if dx else i0[:, 0]
                    yi = i1[:, 1] if dy else i0[:, 1]
                    zi = i1[:, 2] if dz else i0[:, 2]
                    out += weight * self.values[xi, yi, zi]
        return out

    def at_cartesian(self, points) -> np.ndarray:
        return self.at_fractional(self.cell.to_fractional(np.asarray(points)))

    # -- derived fields ----------------------------------------------------
    def difference(self, other: "Grid") -> "Grid":
        """This field minus another, for a difference map.

        The two must be sampled identically; resampling one onto the other
        would hide a mismatch that usually means the files do not belong
        together.
        """
        if self.shape != other.shape:
            raise ValueError(
                f"grids are sampled differently ({self.shape} against "
                f"{other.shape}); a difference map needs the same mesh")
        return Grid(self.values - other.values, self.cell,
                    name=f"{self.name} − {other.name}", units=self.units)

    def scaled(self, factor: float) -> "Grid":
        return Grid(self.values * factor, self.cell, self.name, self.units)


# ---------------------------------------------------------------------------
# marching tetrahedra
# ---------------------------------------------------------------------------

# Six tetrahedra tiling the unit cube, by corner index. Corner c has
# coordinates (c & 1, (c >> 1) & 1, (c >> 2) & 1).
_TETRAHEDRA = np.array([
    [0, 5, 1, 6],
    [0, 1, 2, 6],
    [0, 2, 3, 6],
    [0, 3, 7, 6],
    [0, 7, 4, 6],
    [0, 4, 5, 6],
], int)

# The six edges of a tetrahedron, as pairs of its four vertices.
_TET_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], int)

# For each of the sixteen sign patterns, which edges are cut, as triangles.
# Patterns 0 and 15 are entirely below or above the level and produce nothing.
_TET_TABLE = {
    0: [], 15: [],
    1: [(0, 1, 2)], 14: [(0, 2, 1)],
    2: [(0, 4, 3)], 13: [(0, 3, 4)],
    3: [(1, 2, 4), (1, 4, 3)], 12: [(1, 4, 2), (1, 3, 4)],
    4: [(1, 3, 5)], 11: [(1, 5, 3)],
    5: [(0, 5, 3), (0, 3, 2)], 10: [(0, 3, 5), (0, 2, 3)],
    6: [(0, 4, 5), (0, 5, 1)], 9: [(0, 5, 4), (0, 1, 5)],
    7: [(2, 4, 5)], 8: [(2, 5, 4)],
}


def isosurface(grid: Grid, level: float, step: int = 1,
               cell_range: tuple[int, int, int] = (1, 1, 1)):
    """Triangulate the surface where the field equals `level`.

    Returns ``(vertices, faces, normals)`` in cartesian coordinates, with the
    normals taken from the field's own gradient rather than from the triangles
    -- a gradient normal is smooth across a facet, so the surface does not look
    faceted at the resolution a density grid usually has.

    `step` subsamples the grid, which is the difference between a responsive
    surface and a slow one on a fine mesh.
    """
    values = grid.values[::step, ::step, ::step]
    na, nb, nc = values.shape
    if min(na, nb, nc) < 2:
        return (np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros((0, 3)))

    # wrap one extra plane on each axis so the surface closes across the
    # periodic boundary instead of stopping at the last sample
    padded = np.zeros((na + 1, nb + 1, nc + 1))
    padded[:na, :nb, :nc] = values
    padded[na, :nb, :nc] = values[0, :, :]
    padded[:na, nb, :nc] = values[:, 0, :]
    padded[:na, :nb, nc] = values[:, :, 0]
    padded[na, nb, :nc] = values[0, 0, :]
    padded[na, :nb, nc] = values[0, :, 0]
    padded[:na, nb, nc] = values[:, 0, 0]
    padded[na, nb, nc] = values[0, 0, 0]

    corner_offsets = np.array([[c & 1, (c >> 1) & 1, (c >> 2) & 1]
                               for c in range(8)], int)

    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    cache: dict[tuple, int] = {}

    ii, jj, kk = np.meshgrid(np.arange(na), np.arange(nb), np.arange(nc),
                             indexing="ij")
    base = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)

    # corner values for every cube at once
    cube = np.empty((len(base), 8))
    for c in range(8):
        idx = base + corner_offsets[c]
        cube[:, c] = padded[idx[:, 0], idx[:, 1], idx[:, 2]]

    # only cubes that straddle the level can contribute
    straddles = (cube.min(axis=1) <= level) & (cube.max(axis=1) >= level)
    base = base[straddles]
    cube = cube[straddles]
    if len(base) == 0:
        return (np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros((0, 3)))

    scale = np.array([na, nb, nc], float)

    def vertex_for(origin, ca: int, cb: int, fa: float, fb: float) -> int:
        """Index of the vertex where the level crosses one grid edge.

        Cached by edge rather than by tetrahedron, so adjacent tetrahedra
        share it and the surface comes out watertight instead of as loose
        triangles.
        """
        key = _edge_key(origin, ca, cb, corner_offsets)
        hit = cache.get(key)
        if hit is not None:
            return hit
        denominator = fb - fa
        t = 0.5 if abs(denominator) < 1e-12 else (level - fa) / denominator
        t = float(np.clip(t, 0.0, 1.0))
        pa = origin + corner_offsets[ca]
        pb = origin + corner_offsets[cb]
        hit = len(vertices)
        vertices.append((pa + t * (pb - pa)) / scale)
        cache[key] = hit
        return hit

    above = cube > level
    for tet in _TETRAHEDRA:
        corners = [int(c) for c in tet]
        pattern = (above[:, corners[0]].astype(int)
                   | (above[:, corners[1]].astype(int) << 1)
                   | (above[:, corners[2]].astype(int) << 2)
                   | (above[:, corners[3]].astype(int) << 3))

        for case, triangles in _TET_TABLE.items():
            if not triangles:
                continue
            rows = np.flatnonzero(pattern == case)
            for row in rows:
                origin = base[row]
                for triangle in triangles:
                    face = []
                    for edge in triangle:
                        va, vb = _TET_EDGES[edge]
                        ca, cb = corners[va], corners[vb]
                        face.append(vertex_for(origin, ca, cb,
                                               cube[row, ca], cube[row, cb]))
                    if len(set(face)) == 3:      # skip degenerate triangles
                        faces.append(tuple(face))

    if not vertices:
        return (np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros((0, 3)))

    frac = np.array(vertices, float)
    normals = _gradient_normals(grid, frac)

    nx, ny, nz = (max(1, int(n)) for n in cell_range)
    if (nx, ny, nz) != (1, 1, 1):
        frac, faces, normals = _tile(frac, faces, normals, nx, ny, nz)

    cart = (grid.cell.orth @ frac.T).T
    return cart, np.array(faces, int), normals


def _edge_key(origin, ca: int, cb: int, offsets) -> tuple:
    """A key identifying one grid edge, so neighbouring tetrahedra share a
    vertex instead of each making their own."""
    a = tuple(origin + offsets[ca])
    b = tuple(origin + offsets[cb])
    return (a, b) if a <= b else (b, a)


def _gradient_normals(grid: Grid, frac: np.ndarray) -> np.ndarray:
    """Surface normals from the field gradient, by central differences."""
    delta = 1.0 / np.array(grid.shape, float)
    normals = np.zeros_like(frac)
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = delta[axis]
        forward = grid.at_fractional(frac + step)
        backward = grid.at_fractional(frac - step)
        normals[:, axis] = (forward - backward) / (2 * delta[axis])

    # the gradient is in fractional space; convert to cartesian and point it
    # down the gradient, which is outwards from a high-density region
    inverse = np.linalg.inv(grid.cell.orth)
    normals = -(normals @ inverse)
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths < 1e-12] = 1.0
    return normals / lengths[:, None]


def _tile(frac, faces, normals, nx, ny, nz):
    """Repeat a surface across a block of cells."""
    all_frac, all_faces, all_normals = [], [], []
    base = len(frac)
    n = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                all_frac.append(frac + np.array([i, j, k], float))
                all_normals.append(normals)
                all_faces += [(a + n, b + n, c + n) for a, b, c in faces]
                n += base
    return (np.vstack(all_frac), all_faces, np.vstack(all_normals))


# ---------------------------------------------------------------------------
# 2D sections
# ---------------------------------------------------------------------------

def section(grid: Grid, origin, axis_u, axis_v, size: float = 10.0,
            samples: int = 200):
    """A plane through the field, for a contour map.

    `origin`, `axis_u` and `axis_v` are cartesian. Returns
    ``(values, extent)`` where values is a `samples` x `samples` array and
    extent is ``(umin, umax, vmin, vmax)`` in angstrom, ready to plot.
    """
    u = np.asarray(axis_u, float)
    v = np.asarray(axis_v, float)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        raise ValueError("the plane axes must be non-zero")
    u, v = u / nu, v / nv
    # orthogonalise so the sampled grid is not sheared
    v = v - np.dot(v, u) * u
    nv2 = np.linalg.norm(v)
    if nv2 < 1e-12:
        raise ValueError("the plane axes are parallel")
    v = v / nv2

    half = size / 2.0
    ticks = np.linspace(-half, half, samples)
    uu, vv = np.meshgrid(ticks, ticks, indexing="xy")
    points = (np.asarray(origin, float)
              + uu[..., None] * u + vv[..., None] * v)
    values = grid.at_cartesian(points.reshape(-1, 3)).reshape(samples, samples)
    return values, (-half, half, -half, half)


def miller_plane_axes(cell, h: int, k: int, l: int):
    """Two in-plane axes and the normal of the (hkl) plane, in cartesian.

    The plane normal is the reciprocal-lattice vector h a* + k b* + l c*, which
    is the definition -- not the real-space vector h a + k b + l c, which is
    only the same thing in a cubic cell and is a standard way to get an oblique
    structure's planes wrong.
    """
    orth = np.asarray(cell.orth, float)
    a, b, c = orth[:, 0], orth[:, 1], orth[:, 2]
    volume = float(np.dot(a, np.cross(b, c)))
    if abs(volume) < 1e-12:
        raise ValueError("degenerate cell")
    a_s = np.cross(b, c) / volume
    b_s = np.cross(c, a) / volume
    c_s = np.cross(a, b) / volume

    normal = h * a_s + k * b_s + l * c_s
    length = np.linalg.norm(normal)
    if length < 1e-12:
        raise ValueError("(000) is not a plane")
    normal = normal / length

    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(helper, normal))) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    u = np.cross(helper, normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v, normal


# ---------------------------------------------------------------------------
# bond-valence grids
# ---------------------------------------------------------------------------

def bond_valence_grid(structure: Structure, element: str, ox: int,
                      resolution: float = 0.25,
                      params: bv.ParameterSet | None = None,
                      rmax: float = 6.0) -> Grid:
    """Bond-valence sum a probe cation would have, over the whole cell.

    Vectorised over grid points: the anion positions and their periodic images
    are assembled once, and every point is evaluated against all of them at
    once. The obvious triple loop is about a hundred times slower and makes a
    useful resolution impractical.
    """
    params = params or bv.DEFAULT
    cell = structure.cell
    steps = [max(4, int(round(length / resolution)))
             for length in cell.lengths]

    anions = [a for a in structure.atoms
              if structure.sites[a.site_index].is_anion]
    if not anions:
        raise ValueError("the structure has no anions to bond to")

    # group the anions by element, since each element has its own parameter
    by_element: dict[str, list] = {}
    for atom in anions:
        by_element.setdefault(atom.element, []).append(atom)

    images = _images_within(cell.orth, rmax)
    grid_points = _grid_cartesian(cell, steps)
    total = np.zeros(len(grid_points))
    missing: list[str] = []

    for anion_element, atoms in by_element.items():
        param = params.get(element, ox, anion_element)
        if param is None:
            missing.append(f"{element}-{anion_element}")
            continue
        positions = np.array([a.cart for a in atoms])
        occupancies = np.array([a.occupancy for a in atoms])

        for shift in images @ cell.orth.T:
            shifted = positions + shift
            # (points, anions) distances, in blocks so memory stays bounded
            for start in range(0, len(grid_points), 4096):
                block = grid_points[start:start + 4096]
                d = np.linalg.norm(block[:, None, :] - shifted[None, :, :],
                                   axis=2)
                within = d <= rmax
                if not within.any():
                    continue
                v = np.where(within, np.exp((param.r0 - d) / param.b), 0.0)
                total[start:start + 4096] += (v * occupancies).sum(axis=1)

    grid = Grid(total.reshape(steps), cell,
                name=f"bond valence of {element}{ox:+d}", units="v.u.",
                source=structure.source_path)
    if missing:
        grid.notes.append(
            "no bond-valence parameter for " + ", ".join(sorted(set(missing)))
            + "; those anions contribute nothing to the map")
    grid.notes.append(
        f"probe {element}{ox:+d}; the surface at {abs(ox)} v.u. is where the "
        f"probe would be correctly bonded")
    return grid


def bond_valence_energy(grid: Grid, ox: int, temperature: float = 300.0) -> Grid:
    """Convert a bond-valence map into an energy-like landscape.

    Adams' form: E is proportional to (V - V0)^2, with V0 the probe's formal
    valence. The minimum is where the probe is correctly bonded; the value has
    the shape of an energy but is not one, and is reported in arbitrary units
    rather than eV so that it cannot be mistaken for a computed barrier.
    """
    v0 = float(abs(ox))
    energy = (grid.values - v0) ** 2
    out = Grid(energy, grid.cell,
               name=f"{grid.name} — deviation from {v0:g} v.u.",
               units="(v.u.)²", source=grid.source)
    out.notes = list(grid.notes)
    out.notes.append(
        "a squared deviation from the formal valence, not a calculated "
        "energy; use it to see where a site is plausible, not to quote a "
        "migration barrier")
    return out


def _grid_cartesian(cell, steps) -> np.ndarray:
    ii = np.arange(steps[0]) / steps[0]
    jj = np.arange(steps[1]) / steps[1]
    kk = np.arange(steps[2]) / steps[2]
    fa, fb, fc = np.meshgrid(ii, jj, kk, indexing="ij")
    frac = np.stack([fa.ravel(), fb.ravel(), fc.ravel()], axis=1)
    return (cell.orth @ frac.T).T


def _images_within(orth: np.ndarray, rmax: float) -> np.ndarray:
    """Lattice translations reaching within rmax, by perpendicular width."""
    a, b, c = orth[:, 0], orth[:, 1], orth[:, 2]
    volume = abs(float(np.dot(a, np.cross(b, c))))
    widths = np.array([
        volume / np.linalg.norm(np.cross(b, c)),
        volume / np.linalg.norm(np.cross(c, a)),
        volume / np.linalg.norm(np.cross(a, b)),
    ])
    reps = np.ceil(rmax / widths).astype(int) + 1
    ranges = [np.arange(-r, r + 1) for r in reps]
    grid = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3).astype(float)


# ---------------------------------------------------------------------------
# reading volumetric files
# ---------------------------------------------------------------------------

BOHR = 0.529177210903          # angstrom


def read_chgcar(path):
    """VASP CHGCAR, LOCPOT or ELFCAR. Returns ``(grid, structure)``.

    VASP stores charge as rho times the cell volume, so a CHGCAR's raw numbers
    are divided by the volume to give a density. LOCPOT and ELFCAR are not
    scaled that way, and are told apart by the file name rather than guessed at
    from the values.
    """
    from pathlib import Path

    from .readers import read_poscar

    path = Path(path)
    structure = read_poscar(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    counts_line = 6 if lines[5].split()[0].replace("-", "").isdigit() else 7
    n_atoms = sum(int(x) for x in lines[counts_line].split())
    cursor = counts_line + 2 + n_atoms
    while cursor < len(lines) and not lines[cursor].strip():
        cursor += 1
    if cursor >= len(lines):
        raise ValueError(f"{path.name}: no grid after the coordinates")

    dims = [int(x) for x in lines[cursor].split()[:3]]
    cursor += 1
    wanted = dims[0] * dims[1] * dims[2]

    values = []
    while cursor < len(lines) and len(values) < wanted:
        for token in lines[cursor].split():
            try:
                values.append(float(token))
            except ValueError:
                break
        cursor += 1
    if len(values) < wanted:
        raise ValueError(
            f"{path.name}: expected {wanted} grid values, found {len(values)}")

    # VASP writes the first index fastest
    data = np.array(values[:wanted]).reshape(dims[::-1]).transpose(2, 1, 0)

    name = path.name.upper()
    if "ELFCAR" in name:
        units, label = "", "ELF"
    elif "LOCPOT" in name:
        units, label = "eV", "local potential"
    elif "CHG" in name or "PARCHG" in name:
        data = data / structure.cell.volume
        units, label = "e/A^3", "charge density"
    else:
        units, label = "", "field"

    grid = Grid(data, structure.cell, name=label, units=units, source=str(path))
    if units == "e/A^3":
        grid.notes.append(
            "VASP stores rho times the cell volume, so the values were divided "
            f"by {structure.cell.volume:.2f} A^3 to give a density")
    return grid, structure


def read_cube(path):
    """Gaussian CUBE. Returns ``(grid, structure)``.

    Lengths are in bohr unless the atom count is negative, which is the
    convention for a file already written in angstrom.
    """
    from pathlib import Path

    from .readers import _assemble, cell_from_vectors

    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 7:
        raise ValueError(f"{path.name}: too short to be a CUBE file")

    header = lines[2].split()
    n_atoms = int(header[0])
    origin = np.array([float(x) for x in header[1:4]])
    in_bohr = n_atoms > 0

    vectors, dims = [], []
    for i in range(3):
        parts = lines[3 + i].split()
        n = int(parts[0])
        step = np.array([float(x) for x in parts[1:4]])
        dims.append(abs(n))
        vectors.append(step * abs(n) * (BOHR if in_bohr else 1.0))
    if in_bohr:
        origin = origin * BOHR

    cell = cell_from_vectors(np.array(vectors))
    inverse = np.linalg.inv(cell.orth)

    rows = []
    for i in range(abs(n_atoms)):
        parts = lines[6 + i].split()
        z = int(parts[0])
        position = np.array([float(x) for x in parts[2:5]])
        if in_bohr:
            position = position * BOHR
        symbol = _symbol_for_z(z)
        rows.append((f"{symbol}{i + 1}", symbol,
                     inverse @ (position - origin), 1.0))
    structure = _assemble(path.stem, cell, rows, str(path)) if rows else None

    values = []
    for line in lines[6 + abs(n_atoms):]:
        for token in line.split():
            try:
                values.append(float(token))
            except ValueError:
                pass
    wanted = dims[0] * dims[1] * dims[2]
    if len(values) < wanted:
        raise ValueError(
            f"{path.name}: expected {wanted} grid values, found {len(values)}")
    data = np.array(values[:wanted]).reshape(dims)     # last index fastest
    return Grid(data, cell, name="cube field", source=str(path)), structure


def _symbol_for_z(z: int) -> str:
    for symbol in elements.COLOR:
        if elements.info(symbol).z == z:
            return symbol
    return "X"


def read_xsf(path):
    """XCrySDen XSF, taking the first 3D datagrid. Returns ``(grid, None)``."""
    from pathlib import Path

    from .readers import cell_from_vectors

    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    start = None
    for i, line in enumerate(lines):
        upper = line.upper()
        if "BEGIN_DATAGRID_3D" in upper or "BEGIN_BLOCK_DATAGRID_3D" in upper:
            start = i
            if "BEGIN_DATAGRID_3D" in upper:
                break
    if start is None:
        raise ValueError(f"{path.name}: no 3D datagrid block")

    # skip to the line holding the three dimensions
    cursor = start + 1
    while cursor < len(lines) and len(lines[cursor].split()) != 3:
        cursor += 1
    dims = [int(x) for x in lines[cursor].split()[:3]]
    vectors = np.array([[float(x) for x in lines[cursor + 2 + i].split()[:3]]
                        for i in range(3)])
    cursor = cursor + 5

    values = []
    wanted = dims[0] * dims[1] * dims[2]
    while cursor < len(lines) and len(values) < wanted:
        if "END_DATAGRID" in lines[cursor].upper():
            break
        for token in lines[cursor].split():
            try:
                values.append(float(token))
            except ValueError:
                pass
        cursor += 1
    if len(values) < wanted:
        raise ValueError(
            f"{path.name}: expected {wanted} grid values, found {len(values)}")

    data = np.array(values[:wanted]).reshape(dims[::-1]).transpose(2, 1, 0)

    # An XSF grid repeats the first plane as the last, so that a plotting
    # program can draw a closed cell. Grid treats its data as periodic, so the
    # duplicate has to go or every field is one plane thick too many.
    trimmed = data[:-1, :-1, :-1] if all(n > 1 for n in dims) else data

    cell = cell_from_vectors(vectors)
    grid = Grid(trimmed, cell, name="XSF field", source=str(path))
    grid.notes.append(
        "XSF repeats the first plane as the last; the duplicate was dropped so "
        "the grid is periodic")
    return grid, None


VOLUME_EXTENSIONS = {".cube": read_cube, ".cub": read_cube, ".xsf": read_xsf}

VOLUME_FILE_FILTER = (
    "Volumetric data (CHGCAR* CHG* LOCPOT* ELFCAR* PARCHG* *.cube *.cub *.xsf);;"
    "VASP (CHGCAR* CHG* LOCPOT* ELFCAR* PARCHG*);;"
    "Gaussian CUBE (*.cube *.cub);;"
    "XCrySDen XSF (*.xsf);;"
    "All files (*)"
)


def read_volume(path):
    """Read any supported volumetric file. Returns ``(grid, structure|None)``."""
    from pathlib import Path

    path = Path(path)
    name = path.name.upper()
    if name.startswith(("CHGCAR", "CHG", "LOCPOT", "ELFCAR", "PARCHG")):
        return read_chgcar(path)
    reader = VOLUME_EXTENSIONS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(
            f"{path.name}: not a volumetric format FACET reads. It reads "
            "CHGCAR, LOCPOT, ELFCAR, Gaussian CUBE and XCrySDen XSF.")
    return reader(path)


# ---------------------------------------------------------------------------
# contouring a section
# ---------------------------------------------------------------------------
# Marching squares, the two-dimensional relative of the marching tetrahedra used
# for the isosurfaces. Sixteen cases, of which two are empty; the two saddle
# cases are resolved by the average of the four corners, which is the standard
# disambiguation and the reason a contour map never shows crossing lines.

# For each of the 16 corner sign patterns, the pairs of edges a contour crosses.
# Corners are numbered 0 bottom-left, 1 bottom-right, 2 top-right, 3 top-left;
# edges 0 bottom, 1 right, 2 top, 3 left.
_SQUARE_CASES = {
    0b0000: (), 0b1111: (),
    0b0001: ((3, 0),), 0b1110: ((3, 0),),
    0b0010: ((0, 1),), 0b1101: ((0, 1),),
    0b0100: ((1, 2),), 0b1011: ((1, 2),),
    0b1000: ((2, 3),), 0b0111: ((2, 3),),
    0b0011: ((3, 1),), 0b1100: ((3, 1),),
    0b0110: ((0, 2),), 0b1001: ((0, 2),),
}


def contour_lines(values, extent, levels):
    """Contour line segments through a 2D field.

    ``values`` is the array :func:`section` returns and ``extent`` its
    ``(umin, umax, vmin, vmax)``. Returns ``{level: array of (n, 2, 2)
    segments}`` in the same units as the extent, so the caller draws them
    directly without knowing the grid.

    Segments rather than joined polylines: a contour of a periodic field can
    enter and leave the sampled square many times, and stitching them into
    closed loops would mean deciding which ends belong together -- a decision
    that is wrong exactly where the field is interesting. Drawn as segments the
    picture is identical and nothing is guessed.
    """
    values = np.asarray(values, float)
    if values.ndim != 2 or min(values.shape) < 2:
        return {float(level): np.zeros((0, 2, 2)) for level in np.atleast_1d(levels)}

    umin, umax, vmin, vmax = (float(x) for x in extent)
    rows, cols = values.shape
    # section() builds its grid with indexing="xy", so the first index runs over
    # v and the second over u
    u_axis = np.linspace(umin, umax, cols)
    v_axis = np.linspace(vmin, vmax, rows)

    out = {}
    for level in np.atleast_1d(np.asarray(levels, float)):
        segments = []
        # the four corners of every cell at once
        bl = values[:-1, :-1]
        br = values[:-1, 1:]
        tr = values[1:, 1:]
        tl = values[1:, :-1]
        code = (((bl > level).astype(np.uint8))
                | ((br > level).astype(np.uint8) << 1)
                | ((tr > level).astype(np.uint8) << 2)
                | ((tl > level).astype(np.uint8) << 3))

        for pattern, edges in _SQUARE_CASES.items():
            if not edges:
                continue
            rows_i, cols_i = np.nonzero(code == pattern)
            if not len(rows_i):
                continue
            for j, i in zip(rows_i, cols_i):
                corners = (values[j, i], values[j, i + 1],
                           values[j + 1, i + 1], values[j + 1, i])
                for first, second in edges:
                    a = _edge_point(first, corners, level, u_axis, v_axis, i, j)
                    b = _edge_point(second, corners, level, u_axis, v_axis, i, j)
                    if a is not None and b is not None:
                        segments.append((a, b))

        # the two saddles, resolved by the centre value
        for pattern, alternatives in ((0b0101, ((0, 1), (2, 3))),
                                      (0b1010, ((3, 0), (1, 2)))):
            rows_i, cols_i = np.nonzero(code == pattern)
            for j, i in zip(rows_i, cols_i):
                corners = (values[j, i], values[j, i + 1],
                           values[j + 1, i + 1], values[j + 1, i])
                centre = sum(corners) / 4.0
                if (centre > level) == (pattern == 0b0101):
                    pairs = alternatives
                else:
                    pairs = (((3, 0), (1, 2)) if pattern == 0b0101
                             else ((0, 1), (2, 3)))
                for first, second in pairs:
                    a = _edge_point(first, corners, level, u_axis, v_axis, i, j)
                    b = _edge_point(second, corners, level, u_axis, v_axis, i, j)
                    if a is not None and b is not None:
                        segments.append((a, b))

        out[float(level)] = (np.array(segments, float) if segments
                             else np.zeros((0, 2, 2)))
    return out


def _edge_point(edge: int, corners, level: float, u_axis, v_axis,
                i: int, j: int):
    """Where a contour crosses one edge of a cell, by linear interpolation."""
    # (corner a, corner b) for each edge, in the numbering above
    ends = {0: (0, 1), 1: (1, 2), 2: (2, 3), 3: (3, 0)}[edge]
    va, vb = corners[ends[0]], corners[ends[1]]
    if va == vb:
        return None
    t = (level - va) / (vb - va)
    if not -1e-9 <= t <= 1 + 1e-9:
        return None
    t = min(max(t, 0.0), 1.0)

    # cell corner coordinates
    u0, u1 = u_axis[i], u_axis[i + 1]
    v0, v1 = v_axis[j], v_axis[j + 1]
    positions = {0: (u0, v0), 1: (u1, v0), 2: (u1, v1), 3: (u0, v1)}
    ua, wa = positions[ends[0]]
    ub, wb = positions[ends[1]]
    return (ua + t * (ub - ua), wa + t * (wb - wa))


def nice_levels(values, count: int = 8, log: bool = False):
    """A set of contour levels spanning the data, for a first look.

    Linear by default; logarithmic where the field spans decades, which a
    bond-valence sum or a charge density usually does. Only a starting point --
    the interesting level is rarely a round number.
    """
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(0)
    low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.array([low])
    count = max(int(count), 1)
    if log:
        positive = finite[finite > 0]
        if positive.size:
            low = float(positive.min())
            return np.exp(np.linspace(np.log(low), np.log(high), count + 2))[1:-1]
    return np.linspace(low, high, count + 2)[1:-1]
