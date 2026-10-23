"""The microscopic optical basicity as a field over space, and the three
ways it is drawn: a gradient of translucent level surfaces, a colour-mapped
slice through the 3D view, and either of them isolated to a window of values
or a region.

The basicity of a point
-----------------------
Optical basicity was measured with probe cations (Tl+, Pb2+, Bi3+) sensing
the electron-donor power of the oxygens about them, so the natural field is
the basicity a probe cation would sense at each point r: the mean of the
microscopic basicity lambda_i of the anions it would bond to there,
weighted by the bond valence it would form with each,

    lambda(r) = sum_i s_i(r) lambda_i / sum_i s_i(r),
    s_i(r) = occupancy_i exp((R0 - |r - r_i|) / b)

with R0 and b the bond-valence parameters of the probe-anion pair -- the
same terms a bond-valence map of the probe sums (:func:`facet.core.volume
.bond_valence_grid`), each now carrying the anion's own lambda
(:mod:`facet.core.basicity`). Where the probe would bond to nothing, the
sum of the valences falls below a stated floor and no basicity is drawn:
the field is undefined there rather than extrapolated. An anion without a
lambda (one outside the scale) still bonds the probe but carries no
basicity; the mean is taken over the anions that have one, and the point
is undefined when none does.

This is a construction of the probe picture, stated as such: with the
probe's own bond valences as weights, a point on an anion reads that
anion's lambda, a point between two reads between them, and a point in an
interstice reads what a cation put there would see. It is not a measured
quantity at every point, and it changes with the probe chosen: a smaller
probe (shorter R0) weights the nearest anion more.

Drawing it
----------
A gradient is drawn as level surfaces of the field at several values, each
a translucent surface coloured by the ramp at its level, so a region of
high basicity is wrapped in warm shells and a low one in cool. A slice is
the field sampled on a plane and drawn as colour-mapped cells inside the
3D view. Either can be isolated to a window of values (only what lies
between two levels) or to a region (within a radius of a chosen point, or
inside a slab), by setting the field outside to undefined before meshing.
The meshes are overlay meshes, so every render tier draws them and the
vector exports carry them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from . import basicity as basicity_mod, bv, theme as theme_mod, volume

# The valence the probe has to find for a point to carry a basicity: below
# it the probe is bonded to nothing there. A stated choice; the panels show
# it. 0.2 v.u. is a fifth of a single bond of a monovalent probe.
VALENCE_FLOOR_DEFAULT = 0.2
GRADIENT_LEVELS_DEFAULT = 8
GRADIENT_ALPHA_DEFAULT = 0.22
SLICE_ALPHA_DEFAULT = 0.75
SLICE_BINS_DEFAULT = 24
# The most triangles a whole gradient (every level together) is allowed: a
# finer grid is meshed at a coarser step rather than drawn as millions of
# triangles. Measured on a 12 811-atom glass box of 55 Å at a 0.5 Å grid:
# eight levels at step 1 are 3.1 million triangles (the middle levels cross
# 300 000-380 000 cubes each) and take minutes in the Python marching; at
# step 4 they are 160 000 and take under a second. Marching tetrahedra
# gives about 6.2 triangles per cube the level crosses (the channel
# surfaces' measure).
TRIANGLE_BUDGET = 200_000
TRIANGLES_PER_CUBE = 6.2
MAX_MESH_STEP = 6
# the seconds one (grid point, anion image) pair of the field costs, from
# the channel landscapes' measure on this class of machine, for the
# estimate a panel states before the field runs
PAIRS_PER_S = 7.5e6


@dataclass
class BasicityField:
    """lambda on a grid, with the probe's valence sum beside it."""

    grid: volume.Grid                  # lambda, NaN where undefined
    valence: volume.Grid               # the probe's bond-valence sum
    probe: str
    probe_ox: int
    scale: str
    floor_vu: float
    n_defined: int
    n_points: int
    anions_without_lambda: tuple[str, ...] = ()
    missing_parameters: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.probe}{abs(self.probe_ox)}{'+' if self.probe_ox > 0 else '-'}"

    @property
    def range(self) -> tuple[float, float]:
        """The finite range of lambda; (nan, nan) when nothing is defined."""
        values = self.grid.values[np.isfinite(self.grid.values)]
        if values.size == 0:
            return float("nan"), float("nan")
        return float(values.min()), float(values.max())

    @property
    def fraction_defined(self) -> float:
        return self.n_defined / self.n_points if self.n_points else 0.0

    def legend(self, lo: float | None = None, hi: float | None = None
               ) -> tuple[str, float, float, bool] | None:
        """``(title, lo, hi, diverging)`` for the view's colour bar."""
        a, b = self.range
        lo = a if lo is None else lo
        hi = b if hi is None else hi
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return None
        return (f"basicity λ sensed by {self.label} ({self.scale})",
                float(lo), float(hi), False)


def _finish(total, weighted, covered, floor_vu: float):
    with np.errstate(invalid="ignore", divide="ignore"):
        lam = np.where(covered > 0, weighted / covered, np.nan)
    lam[total < floor_vu] = np.nan
    return lam


# --- a crystal ------------------------------------------------------------------

def basicity_grid(structure, results, scale=basicity_mod.DEFAULT_SCALE, *,
                  probe: str, probe_ox: int, resolution: float = 0.25,
                  params: bv.ParameterSet | None = None, rmax: float = 6.0,
                  floor_vu: float = VALENCE_FLOOR_DEFAULT,
                  v_bond: float | None = None) -> BasicityField:
    """The basicity a probe would sense over a crystal's cell.

    ``results`` is an analysis covering the anions
    (``coordination.analyse_structure(..., cations_only=False)``), from which
    each anion site's lambda is taken at ``v_bond`` (the results' threshold
    when None) on ``scale``. The grid, the summation path and the budget are
    :func:`facet.core.volume.bond_valence_grid`'s.
    """
    params = params or bv.DEFAULT
    scale = basicity_mod.scale_named(scale)
    probe = str(probe)
    probe_ox = int(probe_ox)
    if probe_ox == 0:
        raise ValueError("a probe with oxidation state 0 forms no bond "
                         "valence; give the charge of the probe cation")
    floor = float(floor_vu)
    if not math.isfinite(floor) or floor < 0:
        raise ValueError(f"floor_vu {floor_vu!r}: a valence of 0 or more")
    sb = basicity_mod.structure_basicity(structure, results, scale,
                                         v_bond=v_bond)
    lam_of = {index: m.value for index, m in sb.anions.items()
              if m.defined and np.isfinite(m.value)}
    steps, anions, use_tree = volume.probe_grid_plan(structure, resolution,
                                                     rmax)
    weights = np.array([lam_of.get(a.site_index, np.nan) for a in anions])
    total, weighted, covered, missing = volume._probe_sums(
        structure, anions, probe, probe_ox, steps, params, rmax, use_tree,
        atom_weight=weights)
    lam = _finish(total, weighted, covered, floor)
    cell = structure.cell
    without = sorted({structure.sites[a.site_index].label for a in anions
                      if a.site_index not in lam_of})
    label = f"{probe}{abs(probe_ox)}{'+' if probe_ox > 0 else '-'}"
    grid = volume.Grid(lam.reshape(steps), cell,
                       name=f"basicity sensed by {label}", units="",
                       source=structure.source_path)
    valence = volume.Grid(total.reshape(steps), cell,
                          name=f"bond valence of {label}", units="v.u.",
                          source=structure.source_path)
    out = BasicityField(grid, valence, probe, probe_ox, scale.key, floor,
                        int(np.isfinite(lam).sum()), int(lam.size),
                        tuple(without), tuple(sorted(set(missing))))
    out.notes.append(
        f"lambda sensed by a {label} probe: the mean of the anions' lambda "
        f"weighted by the bond valence the probe would form with each, on "
        f"the {scale.key} scale; undefined where the probe's valence sum is "
        f"below {floor:g} v.u. ({100 * (1 - out.fraction_defined):.1f} % of "
        "the cell)")
    if without:
        out.notes.append("no lambda for " + ", ".join(without)
                         + "; they bond the probe but carry no basicity")
    if missing:
        out.notes.append("no bond-valence parameter for "
                         + ", ".join(sorted(set(missing)))
                         + "; those anions contribute nothing")
    grid.notes = list(out.notes)
    return out


# --- an MD frame ----------------------------------------------------------------

def landscape_seconds(frame, ox_atom, grid_spacing_ang: float,
                      r_cut_ang: float) -> tuple[int, float]:
    """``(grid points, seconds)``: the grid the spacing asks for over the
    frame's box and about how long the field takes, from the anions within
    ``r_cut_ang`` of each point at the box's density and :data:`PAIRS_PER_S`."""
    from . import md_channels

    box = np.asarray(frame.box_ang, np.float64)
    _, n_points, _refusal = md_channels.grid_guard(box, float(grid_spacing_ang),
                                                   None)
    ox = np.asarray(ox_atom, np.int64).reshape(-1)
    anions = int((ox < 0).sum())
    volume = float(frame.volume_ang3)
    per_point = max(1.0, anions / max(volume, 1e-9)
                    * 4.0 / 3.0 * math.pi * float(r_cut_ang) ** 3)
    return int(n_points), float(n_points) * per_point / PAIRS_PER_S


def basicity_landscape(frame, ox_atom, lambda_atom, *, probe: str,
                       probe_ox: int, grid_spacing_ang: float,
                       r_cut_ang: float, params: bv.ParameterSet | None = None,
                       floor_vu: float = VALENCE_FLOOR_DEFAULT,
                       scale: str = basicity_mod.DEFAULT_SCALE,
                       max_grid_points: int | None = None,
                       cancelled: Callable[[], bool] | None = None
                       ) -> BasicityField:
    """The basicity a probe would sense over an MD frame's box.

    ``lambda_atom`` is one lambda per atom of the frame
    (:func:`facet.core.basicity.frame_basicity`; NaN for a cation or an
    anion without one); ``ox_atom`` one oxidation state per atom. The grid
    is :func:`facet.core.md_channels.bv_landscape`'s (its cap applies), the
    contacts those within ``r_cut_ang``. ``cancelled`` is asked once per
    chunk of grid points and stops the sum with
    :class:`facet.core.md_channels.AnalysisCancelled`.
    """
    from scipy.spatial import cKDTree

    from . import bulk, md_channels

    params = params or bv.DEFAULT
    probe = str(probe)
    probe_ox = int(probe_ox)
    if probe_ox == 0:
        raise ValueError("a probe with oxidation state 0 forms no bond "
                         "valence; give the charge of the probe cation")
    h = float(grid_spacing_ang)
    r_cut = float(r_cut_ang)
    if not (math.isfinite(h) and h > 0):
        raise ValueError(f"grid_spacing_ang {grid_spacing_ang!r}: a positive length")
    if not (math.isfinite(r_cut) and r_cut > 0):
        raise ValueError(f"r_cut_ang {r_cut_ang!r}: a positive length")
    floor = float(floor_vu)
    symbols = np.asarray(frame.elements).astype(str)
    ox = np.asarray(ox_atom, np.int64).reshape(-1)
    lam_atom = np.asarray(lambda_atom, np.float64).reshape(-1)
    if ox.shape[0] != symbols.shape[0] or lam_atom.shape[0] != symbols.shape[0]:
        raise ValueError("ox_atom and lambda_atom need one value per atom")
    box = np.asarray(frame.box_ang, np.float64)
    _, n_points, refusal = md_channels.grid_guard(box, h, max_grid_points)
    if refusal:
        raise ValueError(refusal)
    shape = md_channels._grid_shape(box, h)
    is_anion = ox < 0
    if not is_anion.any():
        raise ValueError("the model holds no anion for the probe to bond to")
    anion_elements = sorted(set(symbols[is_anion].tolist()))
    r0 = np.full(symbols.shape[0], np.nan)
    b = np.full(symbols.shape[0], np.nan)
    missing = []
    for element in anion_elements:
        param = params.get(probe, probe_ox, element)
        rows = is_anion & (symbols == element)
        if param is None:
            missing.append(f"{probe}-{element}")
            continue
        r0[rows] = param.r0
        b[rows] = param.b
    atom, _, position = bulk._image_points(frame, r_cut)
    keep = is_anion[atom] & np.isfinite(r0[atom])
    atom, position = atom[keep], position[keep]
    tree = cKDTree(position)
    total = np.zeros(int(n_points))
    weighted = np.zeros(int(n_points))
    covered = np.zeros(int(n_points))
    has = np.isfinite(lam_atom)
    lam_safe = np.where(has, lam_atom, 0.0)
    per_chunk = max(1, min(int(n_points), 200_000))
    for start in range(0, int(n_points), per_chunk):
        if cancelled is not None and cancelled():
            raise md_channels.AnalysisCancelled(
                f"the basicity field was cancelled at grid point {start} "
                f"of {int(n_points)}")
        stop = min(int(n_points), start + per_chunk)
        points = bulk._lattice(md_channels._grid_frac(shape, start, stop), box)
        found = cKDTree(points).sparse_distance_matrix(
            tree, r_cut, output_type="ndarray")
        if not found.shape[0]:
            continue
        rows = atom[found["j"]]
        s = np.exp((r0[rows] - found["v"]) / b[rows])
        n = stop - start
        total[start:stop] = np.bincount(found["i"], weights=s, minlength=n)
        weighted[start:stop] = np.bincount(found["i"], weights=s * lam_safe[rows],
                                           minlength=n)
        covered[start:stop] = np.bincount(found["i"], weights=s * has[rows],
                                          minlength=n)
    lam = _finish(total, weighted, covered, floor)
    label = f"{probe}{abs(probe_ox)}{'+' if probe_ox > 0 else '-'}"
    without = sorted(set(symbols[is_anion & ~has].tolist()))
    notes = [
        f"lambda sensed by a {label} probe: the mean of the anions' lambda "
        f"weighted by the bond valence the probe would form with each "
        f"(contacts to {r_cut:g} Å), on the {scale} scale, on a {h:g} Å grid "
        f"({shape[0]} x {shape[1]} x {shape[2]} points); undefined where the "
        f"probe's valence sum is below {floor:g} v.u. "
        f"({100 * (1 - np.isfinite(lam).mean()):.1f} % of the box)"]
    if without:
        notes.append("no lambda for " + ", ".join(without)
                     + "; they bond the probe but carry no basicity")
    if missing:
        notes.append("no bond-valence parameter for " + ", ".join(missing)
                     + "; those anions contribute nothing")
    grid = md_channels._grid_of(lam.reshape(tuple(int(v) for v in shape)), box,
                                name=f"basicity sensed by {label}", units="",
                                notes=notes)
    valence = md_channels._grid_of(
        total.reshape(tuple(int(v) for v in shape)), box,
        name=f"bond valence of {label}", units="v.u.", notes=notes[:1])
    return BasicityField(grid, valence, probe, probe_ox, str(scale), floor,
                         int(np.isfinite(lam).sum()), int(lam.size),
                         tuple(without), tuple(missing), notes)


# --- isolating ------------------------------------------------------------------

def window_mask(grid: volume.Grid, lo: float, hi: float) -> np.ndarray:
    """True where the field lies within [lo, hi]."""
    v = np.asarray(grid.values)
    with np.errstate(invalid="ignore"):
        return np.isfinite(v) & (v >= float(lo)) & (v <= float(hi))


def sphere_mask(grid: volume.Grid, centre_cart, radius_ang: float) -> np.ndarray:
    """True within ``radius_ang`` of ``centre_cart`` (minimum image)."""
    frac = _grid_fractional(grid)
    centre = np.linalg.inv(np.asarray(grid.cell.orth, float)) @ np.asarray(
        centre_cart, float)
    d = frac - centre[None, :]
    d -= np.round(d)
    cart = (np.asarray(grid.cell.orth, float) @ d.T).T
    return (np.linalg.norm(cart, axis=1) <= float(radius_ang)).reshape(grid.shape)


def slab_mask(grid: volume.Grid, normal_cart, offset_lo: float,
              offset_hi: float) -> np.ndarray:
    """True between the two planes ``n . r = offset_lo`` and ``= offset_hi``
    (``normal_cart`` a unit normal, offsets in Å), on the grid points of the
    home cell."""
    cart = (np.asarray(grid.cell.orth, float) @ _grid_fractional(grid).T).T
    n = np.asarray(normal_cart, float)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    along = cart @ n
    lo, hi = sorted((float(offset_lo), float(offset_hi)))
    return ((along >= lo) & (along <= hi)).reshape(grid.shape)


def axis_slab_mask(grid: volume.Grid, axis: int, centre_fraction: float,
                   thickness_ang: float) -> np.ndarray:
    """True within a slab normal to cell axis ``axis``, centred at a
    fraction of the cell along it, ``thickness_ang`` thick, periodic (the
    MD toolbar's slab)."""
    frac = _grid_fractional(grid)[:, int(axis)]
    length = float(np.linalg.norm(np.asarray(grid.cell.orth, float)[:, int(axis)]))
    half = 0.5 * float(thickness_ang) / max(length, 1e-12)
    d = frac - float(centre_fraction)
    d -= np.round(d)
    return (np.abs(d) <= half + 1e-12).reshape(grid.shape)


def restricted(grid: volume.Grid, keep: np.ndarray) -> volume.Grid:
    """The field with every point outside ``keep`` set to undefined."""
    values = np.array(grid.values, dtype=np.float64)
    values[~np.asarray(keep, bool).reshape(values.shape)] = np.nan
    out = volume.Grid(values, grid.cell, name=grid.name, units=grid.units,
                      source=grid.source)
    out.notes = list(grid.notes)
    return out


def _grid_fractional(grid: volume.Grid) -> np.ndarray:
    na, nb, nc = grid.shape
    ii, jj, kk = np.meshgrid(np.arange(na) / na, np.arange(nb) / nb,
                             np.arange(nc) / nc, indexing="ij")
    return np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)


# --- drawing --------------------------------------------------------------------

def gradient_levels(lo: float, hi: float, n: int) -> list[float]:
    """``n`` levels evenly placed strictly inside [lo, hi], so each has a
    surface; one level at the middle when the range is flat."""
    n = max(1, int(n))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi - lo < 1e-12:
        return [float(lo)] if np.isfinite(lo) else []
    return [lo + (hi - lo) * (k + 0.5) / n for k in range(n)]


def _ramp(colour_of: Callable | None):
    return colour_of if colour_of is not None else (lambda t: theme_mod.ramp(t))


def _finite_normals(normals: np.ndarray) -> np.ndarray:
    out = np.asarray(normals, np.float32)
    bad = ~np.isfinite(out).all(axis=1)
    if bad.any():
        out = out.copy()
        out[bad] = (0.0, 0.0, 1.0)
    return out


def straddling_cubes(values: np.ndarray, level: float, step: int = 1) -> int:
    """How many grid cubes of ``values[::step]`` the level crosses
    (periodic; a cube with an undefined corner crosses nothing), which is
    what the marching tetrahedra will walk."""
    v = np.asarray(values, float)[::step, ::step, ::step]
    if min(v.shape) < 2:
        return 0
    finite = np.isfinite(v)
    inside = finite & (v <= level)
    any_in = inside.copy()
    all_in = inside.copy()
    all_finite = finite.copy()
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                if (dx, dy, dz) == (0, 0, 0):
                    continue
                shift = (-dx, -dy, -dz)
                any_in |= np.roll(inside, shift, axis=(0, 1, 2))
                all_in &= np.roll(inside, shift, axis=(0, 1, 2))
                all_finite &= np.roll(finite, shift, axis=(0, 1, 2))
    return int((any_in & ~all_in & all_finite).sum())


def gradient_plan(values: np.ndarray, levels: Sequence[float], *,
                  budget: int = TRIANGLE_BUDGET,
                  max_step: int = MAX_MESH_STEP) -> tuple[int, int]:
    """``(step, predicted triangles)``: the smallest coarsening of the grid
    at which every level surface together fits the triangle budget, by
    counting the cubes each level crosses (``straddling_cubes`` times
    :data:`TRIANGLES_PER_CUBE`); the largest step when none fits.

    Counting rather than estimating: a basicity field over a glass varies
    at the ångström scale, so a level surface crosses a large share of all
    the cubes -- not the n^(2/3) a smooth field would -- and an estimate
    sent a 55 Å box at 0.5 Å into minutes of marching.
    """
    v = np.asarray(values, float)
    predicted = 0
    for step in range(1, max(1, int(max_step)) + 1):
        if min(v[::step, ::step, ::step].shape) < 2:
            break
        predicted = int(sum(straddling_cubes(v, float(level), step)
                            for level in levels) * TRIANGLES_PER_CUBE)
        if predicted <= budget:
            return step, predicted
    return max(1, min(step, int(max_step))), predicted


def gradient_meshes(grid: volume.Grid, levels: Sequence[float], lo: float,
                    hi: float, *, alpha: float = GRADIENT_ALPHA_DEFAULT,
                    step: int | None = None, colour_of: Callable | None = None,
                    label: str = "basicity",
                    cancelled: Callable[[], bool] | None = None) -> list:
    """One translucent :class:`~facet.gl.vectors.OverlayMesh` per level, in
    the ramp colour of that level over [lo, hi]. ``step`` None takes
    :func:`gradient_plan`'s; ``cancelled`` is asked before each level."""
    from ..gl.vectors import OverlayMesh

    ramp = _ramp(colour_of)
    values = np.asarray(grid.values, float)
    if step is None:
        step, _ = gradient_plan(values, levels)
    out = []
    span = max(float(hi) - float(lo), 1e-12)
    for level in levels:
        if cancelled is not None and cancelled():
            from .md_channels import AnalysisCancelled

            raise AnalysisCancelled("the gradient was cancelled before the "
                                    f"level at {float(level):.4g}")
        vertices, faces, normals = volume.isosurface(grid, float(level),
                                                     step=int(step))
        vertices = np.asarray(vertices, np.float32)
        faces = np.asarray(faces, int)
        if not len(faces) or not len(vertices):
            continue
        flat = faces.reshape(-1)
        tri_normals = (_finite_normals(np.asarray(normals)[flat])
                       if len(normals) == len(vertices)
                       else np.zeros((len(flat), 3), np.float32))
        colour = tuple(float(c) for c in ramp((float(level) - lo) / span))
        out.append(OverlayMesh(vertices[flat], tri_normals, colour,
                               float(alpha), label=label))
    return out


def slice_meshes(grid: volume.Grid, origin, axis_u, axis_v, size: float,
                 samples: int, lo: float, hi: float, *,
                 alpha: float = SLICE_ALPHA_DEFAULT,
                 bins: int = SLICE_BINS_DEFAULT,
                 colour_of: Callable | None = None,
                 label: str = "basicity slice") -> list:
    """The field on a plane, as colour-mapped cells inside the 3D view: one
    :class:`~facet.gl.vectors.OverlayMesh` per colour bin, each holding the
    cells whose value falls in that bin; undefined cells are left out."""
    from ..gl.vectors import OverlayMesh

    ramp = _ramp(colour_of)
    samples = max(2, int(samples))
    values, _extent = volume.section(grid, origin, axis_u, axis_v, size=size,
                                     samples=samples)
    u = np.asarray(axis_u, float)
    v = np.asarray(axis_v, float)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    v = v - np.dot(v, u) * u
    v = v / max(float(np.linalg.norm(v)), 1e-12)
    normal = np.cross(u, v)
    half = float(size) / 2.0
    ticks = np.linspace(-half, half, samples)
    origin = np.asarray(origin, float)
    span = max(float(hi) - float(lo), 1e-12)
    bins = max(1, int(bins))
    # the value of a cell is the mean of its four corners; a cell with an
    # undefined corner is left out rather than guessed
    corners = values
    cell_value = 0.25 * (corners[:-1, :-1] + corners[1:, :-1]
                         + corners[:-1, 1:] + corners[1:, 1:])
    defined = np.isfinite(cell_value)
    with np.errstate(invalid="ignore"):
        t = np.clip((cell_value - float(lo)) / span, 0.0, 1.0)
    t = np.where(defined, t, 0.0)
    which = np.where(defined, np.minimum((t * bins).astype(int), bins - 1), -1)
    out = []
    for k in range(bins):
        rows, cols = np.nonzero(which == k)
        if not rows.size:
            continue
        # values[j, i]: j along v, i along u (volume.section's meshgrid 'xy')
        u0 = ticks[cols]
        u1 = ticks[cols + 1]
        v0 = ticks[rows]
        v1 = ticks[rows + 1]
        p00 = origin + u0[:, None] * u + v0[:, None] * v
        p10 = origin + u1[:, None] * u + v0[:, None] * v
        p01 = origin + u0[:, None] * u + v1[:, None] * v
        p11 = origin + u1[:, None] * u + v1[:, None] * v
        tris = np.concatenate([
            np.stack([p00, p10, p11], axis=1),
            np.stack([p00, p11, p01], axis=1)], axis=0).reshape(-1, 3)
        normals = np.tile(normal.astype(np.float32), (len(tris), 1))
        colour = tuple(float(c) for c in ramp((k + 0.5) / bins))
        out.append(OverlayMesh(tris.astype(np.float32), normals, colour,
                               float(alpha), label=label))
    return out
