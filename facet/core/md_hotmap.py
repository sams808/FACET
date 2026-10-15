"""The modifier hot map of an MD frame: where the network modifiers are
dense, as a smoothed number-density field on a periodic grid and as a
count around each atom.

Two measures, both stated with their parameters:

* ``modifier_counts(frame, elements, r_ang)``: for every atom of the frame,
  the number of atoms of ``elements`` (the modifiers) within ``r_ang`` of
  it, periodic images included and the atom itself left out. An integer
  per atom; the colour mode of the Highlight page draws it.
* ``modifier_density_grid(frame, elements, grid_spacing_ang, sigma_ang)``:
  the number density of those atoms on a grid over the box, each atom
  spread as a normalised Gaussian of width ``sigma_ang`` (periodic, cut at
  four sigma), in atoms per Å³; its integral over the box is the number of
  modifier atoms to within a part in a thousand (the four-sigma cut and
  the grid; the test checks it, and the note states the integral). The surface at a chosen
  multiple of the mean density is what the Highlight page draws: the
  regions where modifiers crowd, the Greaves picture of a modified random
  network, read off the model itself.

No threshold is chosen here. ``elements`` is required: the modifiers are
the cations that are not network formers, which the caller names
(``FrameData.modifiers``); nothing is assumed from the symbols.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import bulk
from .md_model import Frame


@dataclass(frozen=True, eq=False)
class HotMap:
    """The density grid and the numbers the note states."""

    elements: tuple[str, ...]
    grid: object                      # facet.core.volume.Grid, atoms per Å^3
    sigma_ang: float
    grid_spacing_ang: float
    n_atoms: int                      # modifier atoms in the frame
    mean_density: float               # n_atoms / box volume
    max_density: float
    notes: tuple[str, ...] = ()

    def fraction_above(self, level: float) -> float:
        """The fraction of the box where the density exceeds ``level``."""
        values = np.asarray(self.grid.values)
        return float((values > level).mean()) if values.size else 0.0


def _rows_of(frame: Frame, elements) -> np.ndarray:
    wanted = {str(e) for e in elements}
    if not wanted:
        raise ValueError("the hot map needs the elements to count (the "
                         "modifiers); none were named")
    absent = sorted(wanted - set(str(s) for s in frame.species))
    if absent:
        raise ValueError(f"the frame holds no {', '.join(absent)}")
    return np.flatnonzero(np.isin(np.asarray(frame.elements), sorted(wanted)))


def modifier_counts(frame: Frame, elements, r_ang: float) -> np.ndarray:
    """How many atoms of ``elements`` lie within ``r_ang`` of each atom of
    the frame (periodic; the atom itself not counted)."""
    from scipy.spatial import cKDTree

    r = float(r_ang)
    if not math.isfinite(r) or r <= 0:
        raise ValueError(f"r_ang {r_ang!r}: a positive distance in Å")
    rows = _rows_of(frame, elements)
    atom, _, position = bulk._image_points(frame, r)
    sel = np.isin(atom, rows)
    tree = cKDTree(position[sel])
    home = bulk._lattice(frame.frac, frame.box_ang)
    counts = np.asarray([len(h) for h in tree.query_ball_point(home, r)],
                        np.int64)
    # an atom of the counted elements sees itself at distance 0: not a neighbour
    counts[rows] -= 1
    return counts


def modifier_density_grid(frame: Frame, elements, *, grid_spacing_ang: float,
                          sigma_ang: float, max_grid_points: int | None = None
                          ) -> HotMap:
    """The Gaussian-smoothed number density of ``elements`` on a periodic
    grid over the frame (module docstring)."""
    from scipy.spatial import cKDTree
    from . import md_channels

    h = float(grid_spacing_ang)
    sigma = float(sigma_ang)
    if not (math.isfinite(h) and h > 0):
        raise ValueError(f"grid_spacing_ang {grid_spacing_ang!r}: a positive length")
    if not (math.isfinite(sigma) and sigma > 0):
        raise ValueError(f"sigma_ang {sigma_ang!r}: a positive length")
    rows = _rows_of(frame, elements)
    box = np.asarray(frame.box_ang, np.float64)
    _, n_points, refusal = md_channels.grid_guard(box, h, max_grid_points)
    if refusal:
        raise ValueError(refusal)
    shape = md_channels._grid_shape(box, h)
    reach = 4.0 * sigma
    atom, _, position = bulk._image_points(frame, reach)
    sel = np.isin(atom, rows)
    tree = cKDTree(position[sel])
    norm = 1.0 / ((2.0 * math.pi) ** 1.5 * sigma ** 3)
    values = np.zeros(int(n_points))
    per_chunk = max(1, int(2_000_000 // max(1, int(sel.sum()) * 0.05 + 1)))
    per_chunk = min(per_chunk, int(n_points))
    for start in range(0, int(n_points), per_chunk):
        stop = min(int(n_points), start + per_chunk)
        points = bulk._lattice(md_channels._grid_frac(shape, start, stop), box)
        found = cKDTree(points).sparse_distance_matrix(
            tree, reach, output_type="ndarray")
        if found.shape[0]:
            weight = norm * np.exp(-0.5 * (found["v"] / sigma) ** 2)
            values[start:stop] = np.bincount(found["i"], weights=weight,
                                             minlength=stop - start)
    volume = float(frame.volume_ang3)
    mean = rows.size / volume
    grid = md_channels._grid_of(
        values.reshape(tuple(int(v) for v in shape)), box,
        name=f"{'+'.join(sorted(str(e) for e in set(elements)))} density",
        units="atoms/Å^3",
        notes=[f"number density of {', '.join(sorted(set(str(e) for e in elements)))} "
               f"({rows.size} atoms), each spread as a Gaussian of sigma "
               f"{sigma:g} Å cut at {reach:g} Å, on a {h:g} Å grid "
               f"({shape[0]} x {shape[1]} x {shape[2]} points); mean "
               f"{mean:.4g} atoms/Å^3"])
    integral = float(values.sum()) * volume / n_points
    notes = list(grid.notes) + [
        f"the field integrates to {integral:.6g} atoms over the box "
        f"({rows.size} counted); the highest density is "
        f"{float(values.max()):.4g} atoms/Å^3, {float(values.max()) / mean:.3g} "
        "times the mean"]
    return HotMap(elements=tuple(sorted(set(str(e) for e in elements))),
                  grid=grid, sigma_ang=sigma, grid_spacing_ang=h,
                  n_atoms=int(rows.size), mean_density=mean,
                  max_density=float(values.max()) if values.size else 0.0,
                  notes=tuple(notes))
