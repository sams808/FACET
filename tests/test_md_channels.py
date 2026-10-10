"""Channels of an MD model (facet/core/md_channels.py), each result against a
closed form, a law or an independent computation.

What each group pins, and against what:

* **The channel crystal.** A simple cubic framework of O atoms (a = 2.33 Å)
  with one row of atoms removed along a. The bond-valence landscape of a Na+
  probe has, at a threshold between the channel's mismatch (computed here
  by a brute-force sum over images) and the lattice's, exactly one
  accessible region, which spans a and only a, is one-dimensional and
  elongated well beyond 1; the percolation threshold along a lies below
  the channel's mismatch at the removed atoms and the thresholds along b
  and c above the mismatch at the cube centres. The void regions of the
  same crystal: the one region holding the channel spheres spans a only,
  is one-dimensional and elongated; every other region is a single cube
  void of elongation 1 and dimensionality 0; coincident circumcentres of
  the cospherical cubes are merged; a probe too large for the cube voids
  leaves the channel spheres as six isolated regions.
* **Closed forms for voids.** A hand-built sphere set: one sphere is one
  region whose union volume is (4/3) pi r^3 to the grid's precision,
  elongation exactly 1, extent 2r, dimensionality 0; two separated spheres
  are two regions; two overlapping ones are one region with a union below
  the sum and extent d + r_1 + r_2; a sphere overlapping its own periodic
  image spans every axis (dimensionality 3); a chain of spheres crossing
  the box face along a spans a only with the elongation worked out from
  the gyration tensor; the lining atoms of a sphere touching four atoms
  are those four and only those at a lining distance of 0.
* **The bond-valence field against the bulk engine.** On the 3000-atom
  Na2O-3SiO2 model, with a Na atom moved onto a grid point, V at that point
  equals ``bulk.valence_table``'s listed sum for that atom to 1e-10 (the
  same parameters; r_cut chosen so that v > v_list and d < r_cut select the
  same contacts, and checked to lie away from every contact distance).
* **Laws.** The accessible fraction is monotone in Delta; the region
  volumes at a Delta sum to the accessible volume; the percolation
  threshold is a grid value at which the axis is spanned and below which
  (the next lower grid value) it is not; a repulsion exclusion marks the
  grid points within r_excl of a cation and no other (brute force); a
  wall of excluded points across a gives a NaN threshold along a and
  finite ones along b and c; permuting the rows of the 3000-atom model and
  translating it by whole grid steps changes no landscape, modifier or
  void quantity to 1e-10, and an arbitrary translation changes none of the
  sphere-based ones.
* **Modifier counts.** A hand-built O with exactly three Na within the
  cutoff and one beyond counts 3, is rich at k = 3 and not at 4, and forms
  a cluster of four nodes; on a lattice of Na in every cube centre the
  cutoff measured from the first minimum of g_NaO(r) falls in the gap
  between the two shells, every O counts 8 and the one cluster spans all
  three axes; the speciation cross-table separates an NBO from a BO.
* **Exports.** The landscape, region and void grids are
  ``volume.Grid`` objects whose ``at_fractional`` at a grid point returns
  the stored value (the layout is the Grid's).
* **Averages** refuse frames measured with different settings; every
  physical argument is required; no verdict word in any note or string,
  and every public length, radius, cutoff or threshold argument carries
  its unit in its name.

The 3000-atom model is used only by tests that skip when it is absent.
"""
from __future__ import annotations

import ast
import functools
import inspect
import math
import re
from pathlib import Path

import numpy as np
import pytest

from facet.core import bulk, bv, glass, md_model, md_order, volume
from facet.core import md_channels as mc
from facet.core.md_model import frame_from_arrays

ROOT = Path(__file__).resolve().parent.parent
NS3 = Path(r"C:\Users\samso\AppData\Local\Temp\claude\C--Users-samso"
           r"\342b7e11-8171-41a0-8eee-4126254c9cc5\scratchpad"
           r"\facet_md_descriptors\models\ns3_pedone\nvt300K.lammpstrj")
needs_ns3 = pytest.mark.skipif(not NS3.is_file(),
                               reason="the NS3 model is not on this machine")


# ---------------------------------------------------------------------------
# test inputs
# ---------------------------------------------------------------------------

A_ANG = 2.33          # the framework spacing of the channel crystal
N_CELLS = 6


def channel_crystal(n: int = N_CELLS, a: float = A_ANG, remove_row: bool = True):
    """A simple cubic O framework; with ``remove_row`` the atoms at
    y = z = a/2 (one row along a) are missing."""
    points = [(i * a, j * a, k * a) for i in range(n) for j in range(n)
              for k in range(n) if not (remove_row and j == 0 and k == 0)]
    cart = np.array(points, dtype=np.float64) + 0.5 * a
    box = np.diag([n * a] * 3)
    return frame_from_arrays(["O"] * len(points), cart_ang=cart, box_ang=box)


def brute_bvs(frame, points, probe_param, r_cut):
    """The bond-valence sum at Cartesian points: the anions of the frame
    (oxidation state below 0) and their 26 nearest images, summed directly."""
    out = []
    box = frame.box_ang
    anions = frame.cart_ang[ox_of(frame) < 0] - frame.origin_ang
    for x in np.atleast_2d(points):
        total = 0.0
        for shift in np.ndindex(3, 3, 3):
            pos = anions + (np.array(shift) - 1) @ box
            d = np.linalg.norm(pos - x, axis=1)
            total += probe_param.valence(d[d <= r_cut]).sum()
        out.append(total)
    return np.array(out)


def spheres_from(frame, centre_frac, radius_ang, radii=None):
    """A hand-built EmptySpheres: only the fields void_regions reads carry
    meaning (centre_frac, radius_ang, radii_ang, box_volume_ang3, vertices)."""
    centre_frac = np.atleast_2d(np.asarray(centre_frac, dtype=np.float64))
    radius_ang = np.asarray(radius_ang, dtype=np.float64)
    t = centre_frac.shape[0]
    return md_order.EmptySpheres(
        vertices=np.zeros((t, 4), dtype=np.int64),
        vertex_image=np.zeros((t, 4, 3), dtype=np.int64),
        centre_frac=centre_frac, circumradius_ang=radius_ang + 1.0,
        radius_ang=radius_ang, volume_ang3=np.ones(t),
        radii_ang=radii or {s: 1.0 for s in frame.species},
        radii_source="test", margin_ang=1.0,
        volume_sum_ang3=frame.volume_ang3, box_volume_ang3=frame.volume_ang3,
        n_degenerate=0)


def far_frame(box_length: float = 10.0):
    """Two O atoms in a corner of a cubic box, away from the test spheres."""
    cart = np.array([[0.3, 0.3, 0.3], [0.3, 0.3, 1.3]])
    return frame_from_arrays(["O", "O"], cart_ang=cart,
                             box_ang=np.diag([box_length] * 3))


@functools.lru_cache(maxsize=1)
def ns3_frame():
    from facet.core import md_readers

    return md_readers.read_trajectory(NS3, type_map={1: "Si", 2: "O", 3: "Na"}
                                      ).frame(0)


def ox_of(frame):
    return md_model.model_oxidation(frame.species).per_atom(frame.elements)


def moved_copy(frame, shift_frac, rng):
    """The frame translated by ``shift_frac`` (re-wrapped) with its rows
    permuted; returns the frame and the permutation."""
    order = rng.permutation(frame.n_atoms)
    moved = frame_from_arrays(frame.elements[order],
                              frac=frame.frac[order] + np.asarray(shift_frac),
                              box_ang=frame.box_ang)
    return moved, order


# ---------------------------------------------------------------------------
# A. the bond-valence landscape
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def crystal_landscape():
    frame = channel_crystal()
    land = mc.bv_landscape(frame, ox_of(frame), "Na", 1, grid_spacing_ang=0.3,
                           r_cut_ang=6.0)
    return frame, land


def test_channel_crystal_bv_region_percolates_along_a_only(crystal_landscape):
    frame, land = crystal_landscape
    a = A_ANG
    p = bv.DEFAULT.get("Na", 1, "O")
    removed = np.array([[(i + 0.5) * a, 0.5 * a, 0.5 * a] for i in range(N_CELLS)])
    cube = np.array([[3 * a, 3 * a, 3 * a]])
    channel_mismatch = np.abs(brute_bvs(frame, removed, p, 6.0) - 1.0).max()
    lattice_mismatch = np.abs(brute_bvs(frame, cube, p, 6.0) - 1.0).min()
    assert channel_mismatch < 0.25 < 3.0 < lattice_mismatch
    delta = 0.5
    regions = mc.accessible_regions(land, delta)
    assert regions.n_regions == 1
    assert regions.spans.tolist() == [[True, False, False]]
    assert regions.dimensionality.tolist() == [1]
    assert regions.copies.tolist() == [1]
    assert regions.elongation[0] > 3.0
    assert regions.percolates.tolist() == [True, False, False]
    assert regions.mismatch_min_vu[0] == land.mismatch_vu.min()
    thresholds = mc.percolation_thresholds(land)
    assert land.mismatch_vu.min() < thresholds.delta_vu[0] < 0.25
    assert thresholds.delta_vu[1] >= lattice_mismatch
    assert thresholds.delta_vu[2] >= lattice_mismatch
    assert thresholds.delta_any_vu == thresholds.delta_vu[0]
    assert np.isfinite(thresholds.fraction_at).all()
    # the grid holds a point within a quarter step of the removed atoms,
    # so the field there is the brute-force sum to the field's rounding
    i = np.rint((removed - frame.origin_ang) @ np.linalg.inv(frame.box_ang)
                * np.array(land.shape)).astype(int) % np.array(land.shape)
    frac = i / np.array(land.shape)
    exact = brute_bvs(frame, frame.origin_ang + frac @ frame.box_ang, p, 6.0)
    assert np.abs(land.bvs_vu[i[:, 0], i[:, 1], i[:, 2]] - exact).max() < 1e-10


def test_accessible_fraction_is_monotone_and_region_volumes_sum(crystal_landscape):
    frame, land = crystal_landscape
    deltas = np.array([0.0, 0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 200.0])
    fractions = mc.accessible_fraction(land, deltas)
    assert (np.diff(fractions) >= 0).all()
    assert fractions[-1] == 1.0
    assert fractions[0] == 0.0
    for delta in (0.2, 1.0, 5.0):
        regions = mc.accessible_regions(land, delta)
        total = math.fsum(regions.volume_ang3.tolist())
        assert abs(total - regions.accessible_fraction * frame.volume_ang3) \
            < 1e-9 * frame.volume_ang3
        assert abs(regions.accessible_fraction
                   - mc.accessible_fraction(land, delta)) < 1e-15
        assert int(regions.n_points.sum()) == int((regions.label_grid >= 0).sum())
        assert (regions.n_points[:-1] >= regions.n_points[1:]).all()


def test_percolation_threshold_is_the_grid_value_where_spanning_starts(
        crystal_landscape):
    _, land = crystal_landscape
    thresholds = mc.percolation_thresholds(land)
    values = land.sorted_mismatch()
    for axis in range(3):
        delta = thresholds.delta_vu[axis]
        index = int(np.searchsorted(values, delta, side="left"))
        assert values[index] == delta
        assert mc.accessible_regions(land, delta).percolates[axis]
        below = values[index - 1]
        assert below < delta
        assert not mc.accessible_regions(land, below).percolates[axis]
        assert abs(thresholds.fraction_at[axis]
                   - mc.accessible_fraction(land, delta)) < 1e-15


def test_landscape_exports_as_a_grid_in_the_grid_layout(crystal_landscape):
    frame, land = crystal_landscape
    grid = mc.landscape_grid(land, "mismatch")
    assert isinstance(grid, volume.Grid)
    assert grid.shape == land.shape and grid.units == "v.u."
    assert np.array_equal(grid.values, land.mismatch_vu)
    assert abs(grid.cell.volume - frame.volume_ang3) < 1e-9
    rng = np.random.default_rng(5)
    shape = np.array(land.shape)
    for _ in range(20):
        i = rng.integers(0, shape)
        sampled = grid.at_fractional(i / shape)[0]
        assert abs(sampled - land.mismatch_vu[tuple(i)]) < 1e-9
    bvs = mc.landscape_grid(land, "bvs")
    assert np.array_equal(bvs.values, land.bvs_vu)
    regions = mc.accessible_regions(land, 0.5)
    labels = mc.regions_grid(regions)
    assert set(np.unique(labels.values).tolist()) == {0.0, 1.0}
    one = mc.region_indicator_grid(regions, 0)
    assert one.values.sum() == regions.n_points[0]
    with pytest.raises(ValueError, match="no repulsion"):
        mc.landscape_grid(land, "excluded")
    with pytest.raises(ValueError, match="outside"):
        mc.region_indicator_grid(regions, 1)


def test_repulsion_excludes_the_points_near_cations_and_nothing_else():
    frame = channel_crystal(n=4)
    cart = np.vstack([frame.cart_ang, [[1.0, 1.0, 1.0], [5.0, 6.0, 7.0]]])
    symbols = list(frame.elements) + ["Si", "Na"]
    both = frame_from_arrays(symbols, cart_ang=cart, box_ang=frame.box_ang)
    ox = ox_of(both)
    r_excl = 1.3
    land = mc.bv_landscape(both, ox, "Na", 1, grid_spacing_ang=0.4,
                           r_cut_ang=5.0, repulsion=mc.Repulsion(r_excl))
    assert land.repulsion_elements == ("Si",)       # Na is the probe
    shape = np.array(land.shape)
    i, j, k = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    frac = np.stack([i.ravel() / shape[0], j.ravel() / shape[1],
                     k.ravel() / shape[2]], axis=1)
    points = frac @ both.box_ang
    si = both.cart_ang[both.elements == "Si"] - both.origin_ang
    near = np.zeros(points.shape[0], dtype=bool)
    for shift in np.ndindex(3, 3, 3):
        pos = si + (np.array(shift) - 1) @ both.box_ang
        near |= np.linalg.norm(points[:, None, :] - pos[None, :, :],
                               axis=2).min(axis=1) < r_excl
    assert np.array_equal(land.excluded.ravel(), near)
    assert land.n_excluded == int(near.sum()) > 0
    plain = mc.bv_landscape(both, ox, "Na", 1, grid_spacing_ang=0.4,
                            r_cut_ang=5.0)
    assert np.array_equal(plain.bvs_vu, land.bvs_vu)     # the sum is the same
    assert mc.accessible_fraction(land, 1e6) == 1.0 - land.n_excluded / land.n_points
    assert mc.landscape_grid(land, "excluded").values.sum() == land.n_excluded
    named = mc.bv_landscape(both, ox, "Na", 1, grid_spacing_ang=0.4,
                            r_cut_ang=5.0,
                            repulsion=mc.Repulsion(r_excl, {"Na"}))
    assert named.repulsion_elements == ("Na",)
    with pytest.raises(ValueError, match="cations"):
        mc.bv_landscape(both, ox, "Na", 1, grid_spacing_ang=0.4, r_cut_ang=5.0,
                        repulsion=mc.Repulsion(r_excl, {"O"}))


def test_a_wall_of_excluded_points_gives_a_nan_threshold_along_a():
    """Si atoms on a 1 Å grid over the plane x = 5 Å, excluded to 0.8 Å:
    every grid point of the plane is within 0.707 Å of one, so no path
    runs along a, while b and c stay open."""
    o = np.array([[2.0, 2.0, 2.0], [8.0, 7.0, 3.0], [3.0, 8.0, 8.0]])
    si = np.array([[5.0, y, z] for y in range(10) for z in range(10)], float)
    cart = np.vstack([o, si])
    frame = frame_from_arrays(["O"] * 3 + ["Si"] * 100, cart_ang=cart,
                              box_ang=np.diag([10.0, 10.0, 10.0]))
    land = mc.bv_landscape(frame, ox_of(frame), "Na", 1, grid_spacing_ang=0.5,
                           r_cut_ang=4.0, repulsion=mc.Repulsion(0.8))
    thresholds = mc.percolation_thresholds(land)
    assert np.isnan(thresholds.delta_vu[0])
    assert np.isfinite(thresholds.delta_vu[1:]).all()
    assert np.isnan(thresholds.fraction_at[0])
    assert any("closes every path" in n for n in thresholds.notes)
    everything = mc.accessible_regions(land, float(land.mismatch_vu.max()))
    assert everything.percolates.tolist() == [False, True, True]
    averaged = mc.average_percolation_thresholds([thresholds, thresholds])
    assert math.isnan(averaged["a"].mean) and averaged["b"].std == 0.0
    assert any("NaN" in n for n in averaged["a"].notes)


def test_landscape_refusals():
    frame = channel_crystal(n=3)
    ox = ox_of(frame)
    with pytest.raises(TypeError):
        mc.bv_landscape(frame, ox, "Na", 1, r_cut_ang=6.0)        # no spacing
    with pytest.raises(TypeError):
        mc.bv_landscape(frame, ox, "Na", 1, grid_spacing_ang=0.5)  # no r_cut
    with pytest.raises(ValueError, match="above 0"):
        mc.bv_landscape(frame, ox, "Na", 1, grid_spacing_ang=0.0, r_cut_ang=6.0)
    with pytest.raises(ValueError, match="oxidation state 0"):
        mc.bv_landscape(frame, ox, "Na", 0, grid_spacing_ang=0.5, r_cut_ang=6.0)
    with pytest.raises(ValueError, match="whole number"):
        mc.bv_landscape(frame, ox, "Na", 1.0, grid_spacing_ang=0.5, r_cut_ang=6.0)
    cations = frame_from_arrays(["Si"] * 4, frac=np.eye(4, 3) * 0.5,
                                box_ang=np.diag([6.0, 6.0, 6.0]))
    with pytest.raises(ValueError, match="no anion"):
        mc.bv_landscape(cations, ox_of(cations), "Na", 1, grid_spacing_ang=0.5,
                        r_cut_ang=4.0)
    land = mc.bv_landscape(frame, ox, "Na", 1, grid_spacing_ang=0.6,
                           r_cut_ang=5.0)
    with pytest.raises(ValueError, match="0 or more"):
        mc.accessible_fraction(land, [-0.1])
    with pytest.raises(ValueError, match="0 or more"):
        mc.accessible_regions(land, -1.0)
    with pytest.raises(ValueError, match="Landscape"):
        mc.percolation_thresholds(land.mismatch_vu)
    with pytest.raises(ValueError, match="Repulsion"):
        mc.bv_landscape(frame, ox, "Na", 1, grid_spacing_ang=0.6, r_cut_ang=5.0,
                        repulsion=1.0)
    with pytest.raises(ValueError, match="above 0"):
        mc.Repulsion(0.0)
    with pytest.raises(ValueError, match="not the text"):
        mc.Repulsion(1.0, "Si")
    assert all(isinstance(note, str) and note for note in land.notes)
    assert len(land.notes) >= 5


@needs_ns3
def test_grid_point_on_a_na_atom_gives_the_bulk_bvs():
    frame = ns3_frame()
    ox = ox_of(frame)
    k = int(np.flatnonzero(frame.elements == "Na")[7])
    shifted = frame_from_arrays(frame.elements, frac=frame.frac - frame.frac[k],
                                box_ang=frame.box_ang)
    assert shifted.frac[k].tolist() == [0.0, 0.0, 0.0]
    v_list = 1e-4
    p = bv.DEFAULT.get("Na", 1, "O")
    r_cut = p.distance_for(v_list)                       # v > v_list <=> d < r_cut
    land = mc.bv_landscape(shifted, ox, "Na", 1, grid_spacing_ang=1.0,
                           r_cut_ang=r_cut)
    table = bulk.valence_table(shifted, bulk.find_pairs(shifted, r_cut + 0.5),
                               ox, v_list_vu=v_list, r_search_ang=r_cut + 0.5)
    listed = table.d_ang[k, :table.n_listed[k]]
    assert np.abs(listed - r_cut).min() > 1e-6            # no contact on the edge
    bulk_sum = float(table.cum_v_vu[k, table.n_listed[k] - 1])
    assert abs(float(land.bvs_vu[0, 0, 0]) - bulk_sum) < 1e-10
    assert abs(float(land.mismatch_vu[0, 0, 0]) - abs(bulk_sum - 1.0)) < 1e-10


@needs_ns3
def test_landscape_is_invariant_to_permutation_and_whole_step_translation():
    frame = ns3_frame()
    ox = ox_of(frame)
    kw = dict(grid_spacing_ang=0.5, r_cut_ang=5.0)
    land = mc.bv_landscape(frame, ox, "Na", 1, **kw)
    shape = np.array(land.shape)
    rng = np.random.default_rng(1)
    moved, order = moved_copy(frame, np.array([3, 5, 7]) / shape, rng)
    other = mc.bv_landscape(moved, ox[order], "Na", 1, **kw)
    assert other.shape == land.shape
    assert np.abs(np.sort(other.mismatch_vu.ravel())
                  - np.sort(land.mismatch_vu.ravel())).max() < 1e-10
    rolled = np.roll(land.bvs_vu, (3, 5, 7), axis=(0, 1, 2))
    assert np.abs(other.bvs_vu - rolled).max() < 1e-10
    deltas = np.array([0.1, 0.3, 0.6, 1.0])
    assert np.array_equal(mc.accessible_fraction(other, deltas),
                          mc.accessible_fraction(land, deltas))
    one = mc.accessible_regions(land, 0.3)
    two = mc.accessible_regions(other, 0.3)
    assert two.n_regions == one.n_regions
    key = lambda r: np.lexsort((r.mismatch_min_vu, r.volume_ang3))
    for name in ("volume_ang3", "elongation", "mismatch_min_vu"):
        assert np.allclose(getattr(two, name)[key(two)],
                           getattr(one, name)[key(one)], rtol=0.0, atol=1e-10,
                           equal_nan=True)
    assert np.isfinite(one.elongation[one.dimensionality <= 1]).all()
    assert np.isnan(one.elongation[one.dimensionality >= 2]).all()
    assert sorted(map(tuple, two.spans.tolist())) == \
        sorted(map(tuple, one.spans.tolist()))
    # every atom lands in a region of the same size, or in none
    size_one = np.where(one.region_of_atom >= 0,
                        one.volume_ang3[np.maximum(one.region_of_atom, 0)], -1.0)
    size_two = np.where(two.region_of_atom >= 0,
                        two.volume_ang3[np.maximum(two.region_of_atom, 0)], -1.0)
    assert np.abs(size_two - size_one[order]).max() < 1e-10
    assert sum(a.size for a in two.probe_atoms) == sum(a.size for a in one.probe_atoms)
    t_one = mc.percolation_thresholds(land)
    t_two = mc.percolation_thresholds(other)
    assert np.abs(t_two.delta_vu - t_one.delta_vu).max() < 1e-10
    series = mc.average_accessible_fraction([land, other], deltas)
    assert np.array_equal(series.std, np.zeros(4))


# ---------------------------------------------------------------------------
# B. modifier density
# ---------------------------------------------------------------------------

def na_o_frame():
    """One O with three Na at 2.5 Å and one at 4 Å, and a Si far away."""
    cart = np.array([[5.0, 5.0, 5.0], [7.5, 5.0, 5.0], [5.0, 7.5, 5.0],
                     [5.0, 5.0, 7.5], [1.0, 5.0, 5.0], [5.0, 1.0, 1.0]])
    return frame_from_arrays(["O", "Na", "Na", "Na", "Na", "Si"], cart_ang=cart,
                             box_ang=np.diag([12.0, 12.0, 12.0]))


NA_O = dict(modifiers={"Na"}, anions={"O"}, cutoffs_ang={("O", "Na"): 3.0},
            cutoff_sources={("Na", "O"): "test"})


def test_o_with_three_na_counts_three():
    frame = na_o_frame()
    pairs = bulk.find_pairs(frame, 5.0)
    md = mc.modifier_density(frame, pairs, k_rich=3, **NA_O)
    assert md.count.tolist() == [3, -1, -1, -1, -1, -1]
    assert md.anion_count.tolist() == [-1, 1, 1, 1, 0, -1]
    assert md.rich.tolist() == [True] + [False] * 5
    assert md.n_anions == 1 and md.n_rich == 1 and md.fraction_rich == 1.0
    assert md.cluster_sizes.tolist() == [4] and md.cluster_rich.tolist() == [1]
    assert md.label.tolist() == [0, 0, 0, 0, -1, -1]
    assert md.modifiers_in_clusters == 3 and md.n_modifiers == 4
    assert md.count_distribution() == {3: 1}
    assert md.percolates.tolist() == [False, False, False]
    assert md.cluster_dimensionality.tolist() == [0]
    assert md.largest_cluster_fraction == 1.0
    assert md.largest_cluster_rich_fraction == 1.0
    assert md.speciation is None
    assert len(md.as_rows()) == 1
    four = mc.modifier_density(frame, pairs, k_rich=4, **NA_O)
    assert four.n_rich == 0 and four.clusters is None and four.n_clusters == 0
    assert four.label.tolist() == [-1] * 6
    assert four.largest_cluster_rich_fraction == 0.0
    assert any("no anion holds 4" in n for n in four.notes)
    with pytest.raises(ValueError, match="does not join"):
        mc.modifier_density(frame, pairs, modifiers={"Na"}, anions={"O"},
                            cutoffs_ang={("Na", "O"): 3.0, ("Si", "O"): 2.0},
                            cutoff_sources={("Na", "O"): "t", ("Si", "O"): "t"},
                            k_rich=3)
    with pytest.raises(ValueError, match="no cutoff for Na-O"):
        mc.modifier_density(frame, pairs, modifiers={"Na"}, anions={"O"},
                            cutoffs_ang={}, cutoff_sources={}, k_rich=3)
    with pytest.raises(ValueError, match="both"):
        mc.modifier_density(frame, pairs, modifiers={"Na", "O"}, anions={"O"},
                            cutoffs_ang={("Na", "O"): 3.0},
                            cutoff_sources={("Na", "O"): "t"}, k_rich=3)
    with pytest.raises(ValueError, match="at least 1"):
        mc.modifier_density(frame, pairs, k_rich=0, **NA_O)
    with pytest.raises(ValueError, match="no source"):
        mc.modifier_density(frame, pairs, modifiers={"Na"}, anions={"O"},
                            cutoffs_ang={("Na", "O"): 3.0}, cutoff_sources={},
                            k_rich=3)
    with pytest.raises(ValueError, match="beyond"):
        mc.modifier_density(frame, bulk.find_pairs(frame, 2.0), k_rich=3,
                            **NA_O)
    with pytest.raises(ValueError, match="no modifier-anion pair"):
        mc.modifier_density(frame, pairs, modifiers={"K"}, anions={"O"},
                            cutoffs_ang={("K", "O"): 3.0},
                            cutoff_sources={("K", "O"): "t"}, k_rich=3)


def test_measured_cutoff_falls_in_the_gap_and_every_o_counts_eight():
    """O on a simple cubic lattice and Na at every cube centre: the Na-O
    shells are at a sqrt(3)/2 and a sqrt(11)/2, nothing between."""
    a, n = A_ANG, 4
    o = [(i * a, j * a, k * a) for i in range(n) for j in range(n)
         for k in range(n)]
    na = [((i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a) for i in range(n)
          for j in range(n) for k in range(n)]
    frame = frame_from_arrays(["O"] * 64 + ["Na"] * 64, cart_ang=np.array(o + na),
                              box_ang=np.diag([n * a] * 3))
    pairs = bulk.find_pairs(frame, 4.5)
    method = glass.MinimumMethod("valley", None, "midpoint", 0.0)
    measured = mc.measure_cutoffs(frame, pairs, [("Na", "O"), ("O", "Na")],
                                  method, r_max_ang=4.4)
    assert list(measured.cutoffs_ang) == [("Na", "O")]
    cutoff = measured.cutoffs_ang[("Na", "O")]
    assert a * math.sqrt(3) / 2 < cutoff < a * math.sqrt(11) / 2
    assert "first minimum" in measured.cutoff_sources[("Na", "O")]
    assert measured.missing == {} and measured.minima[("Na", "O")].found
    assert measured.r_max_ang == 4.4 and measured.dr_ang == glass.RDF_DR_ANG
    md = mc.modifier_density(frame, pairs, modifiers={"Na"}, anions={"O"},
                             cutoffs_ang=measured.cutoffs_ang,
                             cutoff_sources=measured.cutoff_sources, k_rich=8)
    assert (md.count[frame.elements == "O"] == 8).all()
    assert (md.anion_count[frame.elements == "Na"] == 8).all()
    assert md.n_rich == 64 and md.n_clusters == 1
    assert md.cluster_sizes.tolist() == [128] and md.cluster_rich.tolist() == [64]
    assert md.percolates.tolist() == [True, True, True]
    assert md.cluster_dimensionality.tolist() == [3]
    assert md.largest_cluster_fraction == 1.0
    assert md.count_distribution() == {8: 64}
    nine = mc.modifier_density(frame, pairs, modifiers={"Na"}, anions={"O"},
                               cutoffs_ang=measured.cutoffs_ang,
                               cutoff_sources=measured.cutoff_sources, k_rich=9)
    assert nine.n_rich == 0
    with pytest.raises(ValueError, match="MinimumMethod"):
        mc.measure_cutoffs(frame, pairs, [("Na", "O")], "valley", r_max_ang=4.4)
    with pytest.raises(ValueError, match="holds no K-O pair"):
        mc.measure_cutoffs(frame, pairs, [("K", "O")], method, r_max_ang=4.4)
    avg = mc.average_modifier_density([md, md])
    assert avg["count"].keys == tuple(range(9))
    assert avg["count"].mean.tolist() == [0.0] * 8 + [1.0]
    assert avg["rich fraction"].mean == 1.0 and avg["rich fraction"].std == 0.0
    assert avg["spans a"].mean == 1.0 and avg["clusters"].mean == 1.0
    assert "rich by speciation" not in avg


def test_speciation_cross_table_separates_an_nbo_from_a_bo():
    """O1 bonded to one Si with three Na around it; O2 bonded to two Si with
    one Na: at k = 3 the NBO is modifier-rich and the BO is not."""
    cart = np.array([
        [3.0, 3.0, 3.0],                       # Si1
        [4.6, 3.0, 3.0],                       # O1, 1.6 Å from Si1
        [4.6, 5.4, 3.0], [4.6, 3.0, 5.4], [4.6, 0.6, 3.0],   # Na at 2.4 Å
        [9.0, 9.0, 9.0], [12.2, 9.0, 9.0],     # Si2, Si3
        [10.6, 9.0, 9.0],                      # O2, 1.6 Å from both
        [10.6, 11.4, 9.0]])                    # Na at 2.4 Å from O2
    symbols = ["Si", "O", "Na", "Na", "Na", "Si", "Si", "O", "Na"]
    frame = frame_from_arrays(symbols, cart_ang=cart,
                              box_ang=np.diag([16.0, 16.0, 16.0]))
    ox = ox_of(frame)
    table, _ = bulk.analyse_frame(frame, ox)
    bonds = bulk.bonds_at(table)
    pairs = bulk.find_pairs(frame, 4.0)
    kw = dict(modifiers={"Na"}, anions={"O"}, cutoffs_ang={("Na", "O"): 3.0},
              cutoff_sources={("Na", "O"): "test"}, k_rich=3)
    md = mc.modifier_density(frame, pairs, bonds=bonds, formers={"Si"}, **kw)
    assert md.count[[1, 7]].tolist() == [3, 1]
    assert md.speciation == {"free": (0, 0), "NBO": (1, 1), "BO": (0, 1),
                             "tricluster": (0, 0)}
    assert md.formers == frozenset({"Si"}) and md.v_bond_vu == bonds.v_bond_vu
    assert any("NBO 1 of 1" in n for n in md.notes)
    with pytest.raises(ValueError, match="formers has no default"):
        mc.modifier_density(frame, pairs, bonds=bonds, **kw)
    with pytest.raises(ValueError, match="bulk.Bonds"):
        mc.modifier_density(frame, pairs, bonds=table, formers={"Si"}, **kw)
    plain = mc.modifier_density(frame, pairs, formers={"Si"}, **kw)
    assert plain.speciation is None
    assert any("without bonds" in n for n in plain.notes)
    avg = mc.average_modifier_density([md, md])
    assert avg["rich by speciation"].mean.tolist() == [0.0, 1.0, 0.0, 0.0]
    assert avg["rich fraction among NBO"].mean == 1.0
    assert avg["rich fraction among BO"].mean == 0.0
    assert math.isnan(avg["rich fraction among free"].mean)
    with pytest.raises(ValueError, match="different"):
        mc.average_modifier_density([md, plain])


@needs_ns3
def test_modifier_density_is_invariant_to_permutation_and_translation():
    frame = ns3_frame()
    rng = np.random.default_rng(2)
    moved, order = moved_copy(frame, rng.random(3), rng)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    kw = dict(modifiers={"Na"}, anions={"O"}, cutoffs_ang={("Na", "O"): 3.2},
              cutoff_sources={("Na", "O"): "test"}, k_rich=3)
    one = mc.modifier_density(frame, bulk.find_pairs(frame, 4.0), **kw)
    two = mc.modifier_density(moved, bulk.find_pairs(moved, 4.0), **kw)
    assert np.array_equal(two.count, one.count[order])
    assert np.array_equal(two.anion_count, one.anion_count[order])
    assert np.array_equal(two.rich, one.rich[order])
    assert one.n_rich > 0 and one.n_clusters > 1
    assert sorted(two.cluster_sizes.tolist()) == sorted(one.cluster_sizes.tolist())
    assert sorted(two.cluster_rich.tolist()) == sorted(one.cluster_rich.tolist())
    assert two.percolates.tolist() == one.percolates.tolist()
    assert two.largest_cluster_fraction == one.largest_cluster_fraction
    for c in range(one.n_clusters):
        members = np.flatnonzero(one.label == c)
        assert len(set(two.label[inverse[members]].tolist())) == 1
        assert int(two.label[inverse[members]][0]) >= 0
    assert np.array_equal(two.label[inverse] >= 0, one.label >= 0)


# ---------------------------------------------------------------------------
# C. voids
# ---------------------------------------------------------------------------

VOID_KW = dict(probe_radius_ang=0.0, grid_spacing_ang=0.1, lining_distance_ang=0.0)


def lens_volume(r1, r2, d):
    """The volume common to two spheres of radii r1, r2 at distance d."""
    return (math.pi * (r1 + r2 - d) ** 2
            * (d * d + 2 * d * r2 - 3 * r2 * r2 + 2 * d * r1 + 6 * r1 * r2
               - 3 * r1 * r1) / (12 * d))


def test_one_sphere_is_one_region_with_its_volume_and_elongation_one():
    frame = far_frame(10.0)
    r = 1.5
    spheres = spheres_from(frame, [[0.5, 0.5, 0.5]], [r])
    voids = mc.void_regions(frame, spheres, **VOID_KW)
    exact = 4.0 / 3.0 * math.pi * r ** 3
    assert voids.n_regions == 1 and voids.n_spheres.tolist() == [1]
    assert abs(voids.volume_union_ang3[0] - exact) < 0.02 * exact
    assert abs(voids.volume_sum_ang3[0] - exact) < 1e-12
    assert abs(voids.elongation[0] - 1.0) < 1e-9
    assert abs(voids.extent_ang[0] - 2 * r) < 1e-12
    assert voids.dimensionality.tolist() == [0]
    assert voids.spans.tolist() == [[False, False, False]]
    assert voids.copies.tolist() == [1]
    assert np.abs(voids.centroid_ang[0] - (frame.origin_ang + 5.0)).max() < 1e-12
    assert voids.radius_max_ang[0] == r
    assert voids.lining_atoms[0].size == 0 and (voids.region_of_atom == -1).all()
    assert abs(voids.void_fraction
               - voids.volume_union_ang3[0] / frame.volume_ang3) < 1e-12
    inside = int((voids.label_grid >= 0).sum())
    assert voids.label_grid.max() == 0
    assert abs(inside * voids.voxel_volume_ang3 - voids.volume_union_ang3[0]) < 1e-9
    grid = mc.regions_grid(voids)
    assert isinstance(grid, volume.Grid) and grid.values.sum() == inside
    finer = mc.void_regions(frame, spheres, probe_radius_ang=0.0,
                            grid_spacing_ang=0.05, lining_distance_ang=0.0)
    assert abs(finer.volume_union_ang3[0] - exact) < 0.01 * exact
    assert len(voids.as_rows()) == 1
    assert voids.n_left_out == 0 and voids.n_coincident == 0


def test_two_spheres_separated_overlapping_and_self_overlapping():
    frame = far_frame(10.0)
    two = mc.void_regions(frame, spheres_from(
        frame, [[0.3, 0.5, 0.5], [0.7, 0.5, 0.5]], [1.0, 1.0]), **VOID_KW)
    assert two.n_regions == 2 and two.n_spheres.tolist() == [1, 1]
    r1, r2, d = 1.0, 1.2, 1.5
    pair = spheres_from(frame, [[0.5, 0.5, 0.5], [0.65, 0.5, 0.5]], [r1, r2])
    one = mc.void_regions(frame, pair, **VOID_KW)
    assert one.n_regions == 1 and one.n_spheres.tolist() == [2]
    exact = 4.0 / 3.0 * math.pi * (r1 ** 3 + r2 ** 3) - lens_volume(r1, r2, d)
    assert one.volume_union_ang3[0] < one.volume_sum_ang3[0]
    assert abs(one.volume_union_ang3[0] - exact) < 0.02 * exact
    assert abs(one.extent_ang[0] - (d + r1 + r2)) < 1e-9
    assert one.radius_max_ang[0] == r2
    # a probe of 0.4 Å: probe-centre spheres of 0.6 and 0.8 Å do not meet
    apart = mc.void_regions(frame, pair, probe_radius_ang=0.4,
                            grid_spacing_ang=0.1, lining_distance_ang=0.0)
    assert apart.n_regions == 2 and apart.n_left_out == 0
    # a probe of 1.1 Å: the 1.0 Å sphere is no void for it
    big = mc.void_regions(frame, pair, probe_radius_ang=1.1,
                          grid_spacing_ang=0.1, lining_distance_ang=0.0)
    assert big.n_regions == 1 and big.n_left_out == 1
    assert big.n_spheres.tolist() == [1] and big.radius_max_ang[0] == r2
    # coincident centres are one void
    twin = spheres_from(frame, [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], [1.0, 1.2])
    merged = mc.void_regions(frame, twin, **VOID_KW)
    assert merged.n_regions == 1 and merged.n_spheres.tolist() == [1]
    assert merged.n_coincident == 1 and merged.radius_max_ang[0] == 1.2
    # a sphere wider than half the box overlaps its own images on every axis
    small = frame_from_arrays(["O", "O"], cart_ang=[[0.2, 0.2, 0.2],
                                                    [0.2, 0.2, 1.0]],
                              box_ang=np.diag([4.0, 4.0, 4.0]))
    own = mc.void_regions(small, spheres_from(small, [[0.5, 0.5, 0.5]], [2.5]),
                          probe_radius_ang=0.0, grid_spacing_ang=0.2,
                          lining_distance_ang=0.0)
    assert own.n_regions == 1 and own.dimensionality.tolist() == [3]
    assert own.spans.tolist() == [[True, True, True]]
    assert 0.0 < own.void_fraction < 1.0
    # no sphere reaches the probe: no region, and nothing else
    none = mc.void_regions(frame, pair, probe_radius_ang=5.0,
                           grid_spacing_ang=0.5, lining_distance_ang=0.0)
    assert none.n_regions == 0 and none.n_left_out == 2
    assert none.void_fraction == 0.0 and (none.label_grid == -1).all()
    assert mc.select_regions(none, min_volume_ang3=0.0, min_elongation=0.0).size == 0


def test_chain_of_spheres_across_the_face_spans_a_only():
    frame = far_frame(10.0)
    r = 1.2
    centres = np.array([[x, 0.5, 0.5] for x in (0.1, 0.3, 0.5, 0.7, 0.9)])
    chain = mc.void_regions(frame, spheres_from(frame, centres, [r] * 5),
                            probe_radius_ang=0.0, grid_spacing_ang=0.2,
                            lining_distance_ang=0.0)
    assert chain.n_regions == 1 and chain.n_spheres.tolist() == [5]
    assert chain.dimensionality.tolist() == [1]
    assert chain.spans.tolist() == [[True, False, False]]
    assert chain.copies.tolist() == [1]
    assert chain.percolates.tolist() == [True, False, False]
    # a spanning channel: |T|^2 / 12 along its translation T (one period of
    # 10 Å) and the spheres' own r^2 / 5 across it, whatever the box cut
    expected = math.sqrt((100.0 / 12.0 + r * r / 5.0) / (r * r / 5.0))
    assert abs(chain.elongation[0] - expected) < 1e-9
    assert abs(chain.extent_ang[0] - 10.0) < 1e-9
    assert np.abs(chain.centroid_ang[0][1:] - 5.0).max() < 1e-9
    assert chain.sphere_cell[:, 0].max() - chain.sphere_cell[:, 0].min() == 1
    moved = spheres_from(frame, (centres + 0.37) % 1.0, [r] * 5)
    again = mc.void_regions(frame, moved, probe_radius_ang=0.0,
                            grid_spacing_ang=0.2, lining_distance_ang=0.0)
    for name in ("elongation", "extent_ang", "volume_sum_ang3", "radius_max_ang"):
        assert abs(getattr(again, name)[0] - getattr(chain, name)[0]) < 1e-10
    assert again.spans.tolist() == chain.spans.tolist()
    assert mc.select_regions(chain, min_volume_ang3=10.0, min_elongation=5.0
                             ).tolist() == [0]
    assert mc.select_regions(chain, min_volume_ang3=10.0, min_elongation=6.0
                             ).size == 0


def test_lining_atoms_are_those_touching_the_sphere():
    centre = np.array([5.0, 5.0, 5.0])
    directions = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                          dtype=np.float64) / math.sqrt(3.0)
    atoms = np.vstack([centre + 2.5 * directions, centre + [0.0, 0.0, 3.0]])
    frame = frame_from_arrays(["O"] * 5, cart_ang=atoms,
                              box_ang=np.diag([12.0, 12.0, 12.0]))
    spheres = spheres_from(frame, [centre / 12.0], [1.5], radii={"O": 1.0})
    touching = mc.void_regions(frame, spheres, probe_radius_ang=0.0,
                               grid_spacing_ang=0.2, lining_distance_ang=0.0)
    assert touching.lining_atoms[0].tolist() == [0, 1, 2, 3]
    assert touching.region_of_atom.tolist() == [0, 0, 0, 0, -1]
    near = mc.void_regions(frame, spheres, probe_radius_ang=0.0,
                           grid_spacing_ang=0.2, lining_distance_ang=0.4)
    assert near.lining_atoms[0].tolist() == [0, 1, 2, 3]
    wide = mc.void_regions(frame, spheres, probe_radius_ang=0.0,
                           grid_spacing_ang=0.2, lining_distance_ang=0.6)
    assert wide.lining_atoms[0].tolist() == [0, 1, 2, 3, 4]
    assert wide.region_of_atom.tolist() == [0] * 5
    with pytest.raises(ValueError, match="another frame"):
        mc.void_regions(far_frame(10.0), spheres, **VOID_KW)
    with pytest.raises(ValueError, match="EmptySpheres"):
        mc.void_regions(frame, atoms, **VOID_KW)
    with pytest.raises(TypeError):
        mc.void_regions(frame, spheres, probe_radius_ang=0.0,
                        grid_spacing_ang=0.2)


def test_channel_crystal_void_region_percolates_along_a_only():
    frame = channel_crystal()
    spheres = md_order.empty_spheres(frame, {"O": 1.0}, radii_source="test")
    voids = mc.void_regions(frame, spheres, probe_radius_ang=0.0,
                            grid_spacing_ang=0.3, lining_distance_ang=0.1)
    a = A_ANG
    assert voids.n_coincident > 0
    assert voids.n_regions == 1 + N_CELLS ** 3 - 24
    assert voids.n_spheres[0] == 6 + 24           # the channel and its sheath
    assert voids.spans[0].tolist() == [True, False, False]
    assert voids.dimensionality[0] == 1 and voids.copies[0] == 1
    assert voids.elongation[0] > 3.0
    assert abs(voids.radius_max_ang[0] - (a * math.sqrt(5) / 2 - 1.0)) < 1e-9
    assert not voids.spans[1:].any() and (voids.dimensionality[1:] == 0).all()
    assert (voids.n_spheres[1:] == 1).all()
    assert np.abs(voids.elongation[1:] - 1.0).max() < 1e-9
    assert np.abs(voids.radius_max_ang[1:] - (a * math.sqrt(3) / 2 - 1.0)).max() \
        < 1e-9
    assert voids.percolates.tolist() == [True, False, False]
    assert mc.select_regions(voids, min_volume_ang3=50.0, min_elongation=2.0
                             ).tolist() == [0]
    assert mc.select_regions(voids, min_volume_ang3=0.0, min_elongation=0.0
                             ).size == voids.n_regions
    assert abs(voids.volume_union_ang3.sum() / frame.volume_ang3
               - voids.void_fraction) < 1e-12
    assert voids.void_fraction < 1.0
    assert voids.lining_atoms[0].size >= 24
    assert int((voids.label_grid >= 0).sum()) == int(np.rint(
        voids.volume_union_ang3.sum() / voids.voxel_volume_ang3))
    cube = 4.0 / 3.0 * math.pi * (a * math.sqrt(3) / 2 - 1.0) ** 3
    assert np.abs(voids.volume_union_ang3[1:] - cube).max() < 0.15 * cube
    big = mc.void_regions(frame, spheres, probe_radius_ang=1.1,
                          grid_spacing_ang=0.3, lining_distance_ang=0.1)
    assert big.n_regions == 6 and (big.dimensionality == 0).all()
    assert big.n_left_out == len(spheres) - 36
    assert (big.n_spheres == 1).all()
    avg = mc.average_void_regions([voids, voids],
                                  volume_edges_ang3=[0.0, 10.0, 100.0, 1000.0],
                                  elongation_edges=[0.0, 1.5, 3.0, 50.0])
    assert avg["void fraction"].mean == voids.void_fraction
    assert avg["regions"].mean == voids.n_regions
    assert avg["spanning regions"].mean == 1.0
    assert avg["region volume"].mean.tolist() == [voids.n_regions - 1, 0.0, 1.0]
    assert avg["region elongation"].mean.tolist() == [voids.n_regions - 1, 0.0, 1.0]
    with pytest.raises(ValueError, match="different probe"):
        mc.average_void_regions([voids, big],
                                volume_edges_ang3=[0.0, 10.0, 1000.0],
                                elongation_edges=[0.0, 2.0, 50.0])


@needs_ns3
def test_void_regions_are_invariant_to_permutation_and_translation():
    frame = ns3_frame()
    radii = {s: round(0.5 * v, 2) for s, v in md_order.vdw_radii_ang(frame).items()}
    rng = np.random.default_rng(4)
    moved, order = moved_copy(frame, rng.random(3), rng)
    kw = dict(probe_radius_ang=0.3, grid_spacing_ang=0.5, lining_distance_ang=0.3)
    one = mc.void_regions(frame, md_order.empty_spheres(
        frame, radii, radii_source="test"), **kw)
    two = mc.void_regions(moved, md_order.empty_spheres(
        moved, radii, radii_source="test"), **kw)
    assert one.n_regions > 1 and two.n_regions == one.n_regions
    key = lambda v: np.lexsort((v.elongation, v.volume_sum_ang3))
    for name, tol in (("volume_sum_ang3", 1e-9), ("elongation", 1e-10),
                      ("extent_ang", 1e-9), ("radius_max_ang", 1e-10)):
        assert np.allclose(getattr(two, name)[key(two)],
                           getattr(one, name)[key(one)], rtol=0.0, atol=tol,
                           equal_nan=True)
    assert np.isfinite(one.elongation[one.dimensionality <= 1]).all()
    assert np.array_equal(two.n_spheres[key(two)], one.n_spheres[key(one)])
    assert np.array_equal(two.dimensionality[key(two)],
                          one.dimensionality[key(one)])
    assert two.percolates.tolist() == one.percolates.tolist()
    assert sorted(a.size for a in two.lining_atoms) == \
        sorted(a.size for a in one.lining_atoms)
    assert int((two.region_of_atom >= 0).sum()) == int((one.region_of_atom >= 0).sum())
    # the union volumes too, under a translation by whole grid steps
    shape = np.array(one.shape)
    stepped, order2 = moved_copy(frame, np.array([2, 3, 4]) / shape, rng)
    three = mc.void_regions(stepped, md_order.empty_spheres(
        stepped, radii, radii_source="test"), **kw)
    assert three.shape == one.shape
    assert np.abs(three.volume_union_ang3[key(three)]
                  - one.volume_union_ang3[key(one)]).max() < 1e-9
    assert abs(three.void_fraction - one.void_fraction) < 1e-12


# ---------------------------------------------------------------------------
# common: selection, averages, no verdicts, units, no Qt
# ---------------------------------------------------------------------------

def test_select_regions_on_both_kinds_requires_both_thresholds(crystal_landscape):
    _, land = crystal_landscape
    regions = mc.accessible_regions(land, 0.5)
    assert mc.select_regions(regions, min_volume_ang3=1.0, min_elongation=2.0
                             ).tolist() == [0]
    assert mc.select_regions(regions, min_volume_ang3=1e9, min_elongation=0.0
                             ).size == 0
    with pytest.raises(TypeError):
        mc.select_regions(regions, min_volume_ang3=1.0)
    with pytest.raises(ValueError, match="0 or more"):
        mc.select_regions(regions, min_volume_ang3=-1.0, min_elongation=0.0)
    with pytest.raises(ValueError, match="results"):
        mc.select_regions(land, min_volume_ang3=0.0, min_elongation=0.0)
    assert len(regions.as_rows()) == 1
    assert regions.as_rows()[0]["spans a"] is True


def test_averages_refuse_mixed_settings_and_report_zero_spread(crystal_landscape):
    frame, land = crystal_landscape
    other = mc.bv_landscape(frame, ox_of(frame), "Na", 1, grid_spacing_ang=0.3,
                            r_cut_ang=5.0)
    with pytest.raises(ValueError, match="different"):
        mc.average_accessible_fraction([land, other], [0.1, 0.5])
    with pytest.raises(ValueError, match="1-D"):
        mc.average_accessible_fraction([land, land], 0.5)
    series = mc.average_accessible_fraction([land, land], [0.1, 0.5, 1.0],
                                            frames=[3, 7])
    assert series.n_frames == 2 and np.array_equal(series.std, np.zeros(3))
    assert series.axis_name == "delta_vu"
    assert np.array_equal(series.mean, mc.accessible_fraction(land, [0.1, 0.5, 1.0]))
    thresholds = mc.percolation_thresholds(land)
    scalars = mc.average_percolation_thresholds([thresholds, thresholds])
    assert set(scalars) == {"a", "b", "c", "any"}
    assert scalars["a"].mean == thresholds.delta_vu[0] and scalars["a"].std == 0.0
    assert scalars["any"].mean == thresholds.delta_any_vu
    with pytest.raises(ValueError, match="different"):
        mc.average_percolation_thresholds([thresholds,
                                           mc.percolation_thresholds(other)])
    with pytest.raises(ValueError, match="one Landscape"):
        mc.average_accessible_fraction([thresholds], [0.1])
    assert len(thresholds.as_rows()) == 3


def collect_notes():
    frame = channel_crystal(n=4)
    ox = ox_of(frame)
    land = mc.bv_landscape(frame, ox, "Na", 1, grid_spacing_ang=0.4,
                           r_cut_ang=5.0, repulsion=mc.Repulsion(0.5))
    notes = list(land.notes)
    regions = mc.accessible_regions(land, 0.5)
    notes += regions.notes
    notes += mc.percolation_thresholds(land).notes
    spheres = md_order.empty_spheres(frame, {"O": 1.0}, radii_source="test")
    notes += mc.void_regions(frame, spheres, probe_radius_ang=0.0,
                             grid_spacing_ang=0.4, lining_distance_ang=0.1).notes
    small = na_o_frame()
    notes += mc.modifier_density(small, bulk.find_pairs(small, 5.0), k_rich=3,
                                 **NA_O).notes
    notes += mc.landscape_grid(land).notes
    return notes


VERDICT = re.compile(
    r"\b(good|bad|poor|excellent|acceptable|unacceptable|correct|incorrect|"
    r"wrong|reliable|unreliable|trustworthy|untrustworthy|unusable|should|"
    r"proves|confirms)\b", re.IGNORECASE)


def test_no_note_carries_a_verdict():
    notes = collect_notes()
    assert len(notes) >= 20
    assert [n for n in notes if VERDICT.search(n)] == []


def test_no_string_in_the_module_carries_a_verdict():
    source = (ROOT / "facet" / "core" / "md_channels.py").read_text(
        encoding="utf-8")
    strings = [node.value for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert len(strings) > 50
    assert [s for s in strings if VERDICT.search(s)] == []
    comments = [line for line in source.splitlines() if "#" in line]
    assert [c for c in comments if VERDICT.search(c.split("#", 1)[1])] == []


def test_public_arguments_carry_their_unit():
    bare = {"r", "d", "x", "h", "radius", "radii", "cutoff", "cutoffs",
            "margin", "spacing", "probe_radius", "delta", "deltas", "edges",
            "distance", "volume", "r_cut", "r_excl", "lining", "min_volume",
            "grid_spacing", "threshold", "thresholds"}
    found = []
    for name in mc.__all__:
        obj = getattr(mc, name)
        if not inspect.isfunction(obj):
            continue
        for p in inspect.signature(obj).parameters:
            if p in bare:
                found.append(f"{name}({p})")
    assert found == []
    assert {"grid_spacing_ang", "r_cut_ang"} <= set(
        inspect.signature(mc.bv_landscape).parameters)
    assert "delta_vu" in inspect.signature(mc.accessible_regions).parameters
    assert {"probe_radius_ang", "grid_spacing_ang", "lining_distance_ang"} <= set(
        inspect.signature(mc.void_regions).parameters)
    assert {"min_volume_ang3", "min_elongation"} <= set(
        inspect.signature(mc.select_regions).parameters)
    assert "r_excl_ang" in inspect.signature(mc.Repulsion).parameters


def test_nothing_physical_has_a_default():
    for function, names in ((mc.bv_landscape, ("grid_spacing_ang", "r_cut_ang")),
                            (mc.void_regions, ("probe_radius_ang",
                                               "grid_spacing_ang",
                                               "lining_distance_ang")),
                            (mc.modifier_density, ("cutoffs_ang",
                                                   "cutoff_sources", "k_rich")),
                            (mc.select_regions, ("min_volume_ang3",
                                                 "min_elongation")),
                            (mc.measure_cutoffs, ("r_max_ang",)),
                            (mc.average_void_regions, ("volume_edges_ang3",
                                                       "elongation_edges"))):
        params = inspect.signature(function).parameters
        for name in names:
            assert params[name].default is inspect.Parameter.empty, \
                f"{function.__name__}({name}) has a default"


def test_the_module_imports_without_qt():
    import subprocess
    import sys

    code = ("import sys, facet.core.md_channels;"
            "print('QT' if any(m.startswith(('PySide6', 'matplotlib')) "
            "for m in sys.modules) else 'CLEAN')")
    result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout


def test_sheared_box_gives_the_brute_force_field_and_the_sphere_volume():
    """A strongly tilted cell: the grid shape, the image enumeration and the
    stencil all follow the cell vectors; the field at grid points equals a
    brute-force sum over 27 images, and a sphere on the sheared grid still
    has its volume and elongation 1."""
    box = np.array([[9.0, 0.0, 0.0], [3.1, 8.2, 0.0], [-2.4, 2.9, 10.5]])
    rng = np.random.default_rng(11)
    frame = frame_from_arrays(["O"] * 30 + ["Si"] * 10,
                              frac=rng.uniform(0, 1, (40, 3)), box_ang=box)
    land = mc.bv_landscape(frame, ox_of(frame), "Na", 1, grid_spacing_ang=0.7,
                           r_cut_ang=5.0)
    shape = np.array(land.shape)
    assert shape.tolist() == [math.ceil(np.linalg.norm(v) / 0.7) for v in box]
    p = bv.DEFAULT.get("Na", 1, "O")
    for _ in range(15):
        i = rng.integers(0, shape)
        x = frame.origin_ang + (i / shape) @ box
        assert abs(land.bvs_vu[tuple(i)] - brute_bvs(frame, x, p, 5.0)[0]) < 1e-10
    regions = mc.accessible_regions(land, 0.5)
    assert regions.n_regions >= 1
    assert abs(math.fsum(regions.volume_ang3.tolist())
               - regions.accessible_fraction * frame.volume_ang3) < 1e-9
    small = frame_from_arrays(["O", "O"], frac=[[0.02, 0.02, 0.02],
                                                [0.05, 0.02, 0.02]], box_ang=box)
    r = 1.5
    voids = mc.void_regions(small, spheres_from(small, [[0.5, 0.5, 0.5]], [r]),
                            probe_radius_ang=0.0, grid_spacing_ang=0.1,
                            lining_distance_ang=0.0)
    exact = 4.0 / 3.0 * math.pi * r ** 3
    assert voids.n_regions == 1
    assert abs(voids.volume_union_ang3[0] - exact) < 0.02 * exact
    assert abs(voids.elongation[0] - 1.0) < 1e-9
    assert voids.dimensionality.tolist() == [0]
    assert np.abs(voids.centroid_ang[0] - np.array([0.5, 0.5, 0.5]) @ box).max() \
        < 1e-9
    assert abs(mc.regions_grid(voids).cell.volume - small.volume_ang3) < 1e-9


# ---------------------------------------------------------------------------
# G. cancellation and the grid cap
# ---------------------------------------------------------------------------

def test_the_landscape_stops_when_the_hook_flips_after_the_first_chunk(
        monkeypatch):
    """One grid point a chunk (PAIRS_PER_CHUNK patched to 1), a hook that is
    False once and then True: the sum raises AnalysisCancelled at the second
    chunk, well under a second, instead of walking the grid."""
    import time

    monkeypatch.setattr(mc, "PAIRS_PER_CHUNK", 1)
    frame = na_o_frame()
    answers = iter([False] + [True] * 1_000_000)
    clock = time.perf_counter()
    with pytest.raises(mc.AnalysisCancelled, match="cancelled"):
        mc.bv_landscape(frame, ox_of(frame), "Na", 1, grid_spacing_ang=0.4,
                        r_cut_ang=4.0, cancelled=lambda: next(answers))
    assert time.perf_counter() - clock < 1.0


def test_regions_thresholds_and_voids_ask_the_cancel_hook():
    frame = na_o_frame()
    land = mc.bv_landscape(frame, ox_of(frame), "Na", 1, grid_spacing_ang=1.0,
                           r_cut_ang=4.0)
    with pytest.raises(mc.AnalysisCancelled):
        mc.accessible_regions(land, 0.3, cancelled=lambda: True)
    with pytest.raises(mc.AnalysisCancelled):
        mc.percolation_thresholds(land, cancelled=lambda: True)
    with pytest.raises(mc.AnalysisCancelled):
        mc.void_regions(frame, spheres_from(frame, [[0.5, 0.5, 0.5]], [1.5]),
                        probe_radius_ang=0.0, grid_spacing_ang=0.5,
                        lining_distance_ang=0.0, cancelled=lambda: True)
    # a hook that never says stop changes nothing
    same = mc.bv_landscape(frame, ox_of(frame), "Na", 1, grid_spacing_ang=1.0,
                           r_cut_ang=4.0, cancelled=lambda: False)
    assert np.array_equal(same.bvs_vu, land.bvs_vu)
    # the analysis layer raises this very class (md_analysis aliases it)
    from facet.core import md_analysis

    assert md_analysis.AnalysisCancelled is mc.AnalysisCancelled


def test_a_grid_beyond_the_cap_is_refused_with_the_numbers():
    """The 12 Å box at 0.5 Å asks for 24 x 24 x 24 = 13 824 points: a cap of
    1 000 refuses it before anything is allocated, naming the counts and a
    spacing that fits; the default cap admits it."""
    frame = na_o_frame()
    with pytest.raises(ValueError) as err:
        mc.bv_landscape(frame, ox_of(frame), "Na", 1, grid_spacing_ang=0.5,
                        r_cut_ang=4.0, max_grid_points=1000)
    text = str(err.value)
    assert "24 x 24 x 24" in text and "13,824" in text and "1,000" in text
    assert "0.5" in text and "would fit" in text
    with pytest.raises(ValueError, match="grid points"):
        mc.void_regions(frame, spheres_from(frame, [[0.5, 0.5, 0.5]], [1.5]),
                        probe_radius_ang=0.0, grid_spacing_ang=0.5,
                        lining_distance_ang=0.0, max_grid_points=1000)
    shape, n_points, refusal = mc.grid_guard(frame.box_ang, 0.5, 1000)
    assert shape == (24, 24, 24) and n_points == 13824
    assert "13,824" in refusal
    # the spacing suggested is the cube root of volume over the cap
    assert f"{(frame.volume_ang3 / 1000) ** (1 / 3):.2g}" in refusal
    assert mc.grid_guard(frame.box_ang, 0.5)[2] == ""
    assert mc.MAX_GRID_POINTS == 20_000_000
