"""Coordination of every atom of an MD frame at once.

The crystal path analyses one site at a time: a ``NeighborFinder`` query, a
list of ``ContactRow`` objects, one ``bv.ParameterSet.get`` per contact, then
the polyhedron geometry. For a 20-site crystal that costs little. For a
10 000-atom glass, where every atom is its own site, it is a Python loop over
10 000 sites, and Step 0 of the MD work measured it at 17-28 s per frame on
this machine even with its two quadratic hot spots patched out. This module
computes the same quantities for all atoms with one neighbour search and array
operations: 0.5-0.6 s for a 10 000-atom frame on the same machine pinned to
its P-cores, 0.6-1.1 s unpinned as an ordinary script runs (the table below),
so a trajectory of a hundred frames takes minutes, not hours.

WHAT IS COMPUTED, AND WHY IT EQUALS THE CRYSTAL PATH
----------------------------------------------------
The quantities are those of :class:`facet.core.coordination.SiteResult`, field
for field: CN(v_bond), CN(v_list), the bond-valence sum over each set, the
bond-valence vector and phi over each set, and the width of the plateau the
threshold sits on [1, 2]. They are computed so that a crystal unrolled into a
supercell gives, atom by atom, what ``coordination.analyse_site`` gives for the
atom each copy came from; ``tests/test_bulk_equivalence.py`` checks that to
1e-10 on six bundled and five reference crystals. Each rule mirrors a line of
the crystal code:

* **Who counts as a counter-ion.** An atom is an anion when its oxidation state
  is negative, which is ``Site.is_anion`` with a state set
  (``structure.py:59-63``); a cation (state 0 included) lists anions and an
  anion lists cations (``coordination.py:242-247``). An atom with no state is
  refused: ``Site.is_anion`` would fall back on ``elements.is_anion_like``,
  which is True for B, P, Bi, Ge and Pb.
* **The same parameter.** One ``params.get(cation, cation_ox, anion)`` per
  combination present, in the argument order ``analyse_site`` uses from both
  ends (``coordination.py:255-261``), so overrides, Table 2 and the estimator
  answer as they do there. The valence is ``BVParam.valence`` called once per
  parameter on the distance array (``bv.py:67-69``); Step 0 measured that
  bit-identical to the per-contact scalar calls (0 differences in 5 000).
* **The same contacts.** The search radius follows
  ``neighbors.search_radius_for`` (``neighbors.py:220-237``), taken over every
  (element, state) cation species present rather than the last state of each
  element its dict keeps (``:229-230``), so it is never smaller. Periodic
  images are the translated copies ``NeighborFinder`` builds
  (``neighbors.py:99-110``), less those farther than r from every point of
  the box, which can pair with no atom in it (:func:`_image_points`); the
  pair set is the same. Contacts at d <= 0.4 Å are left out as there
  (``:89-90``, ``:128-131``), and here they are also counted
  (:attr:`PairTable.n_below_d_min`) and noted.
* **The same listing and the same thresholds.** A contact is listed when
  v > v_list and bonded when v > v_bond, both strict (``coordination.py:291-292``,
  ``:322-323``); the sums and phi are NaN for an atom with no listed or no
  bonded contact (``:116-125``, ``:329-344``). A contact with no parameter
  carries no valence, is in no sum, and is counted per atom and per pair,
  never given zero (``:263-268``).
* **The plateau rule of the code, not of its docstring.**
  ``SiteResult.plateau_decades`` is documented as NaN "exactly on a step edge"
  (``coordination.py:172-173``), but ``Plateau.contains`` is
  ``v_low < v <= v_high`` (``:81-82``), so at v_bond equal to the k-th valence
  it reports plateau k while CN, with its strict ``>``, is k-1. This module
  reproduces the code: with the listed valences sorted in descending order,
  k = #(v >= v_bond), the width is log10(v_k / v_(k+1)) with v_list standing
  below the last contact (``:394-421``), and it is NaN when k = 0, or when
  k is the whole list and v_bond <= v_list. Infinite when the lower edge is not
  positive (``:71-74``).

Two choices differ from the crystal code. Neither changes a number at the
stated tolerance; the last paragraph of this section gives the one place
where a last-bit difference can still show:

* **One valence per bond, identical from both ends, from the fractions.**
  ``NeighborFinder`` computes a contact vector as (image position) - centre;
  seen from the other atom, the rounding differs in the last bit, and a
  contact whose valence lies within a rounding of v_list could then be listed
  from one end and not the other. Here the vector from i to the image n of j
  is ``(f_j - f_i + n) . (a, b, c)``, formed term by term from the fractions
  (:func:`_pair_geometry`), which is exactly the negative of the vector from j
  to the image -n of i, so the distance and the valence are bit-identical
  from both ends, and the bonds read from the cation rows and from the anion
  rows are the same set (``tests/test_bulk.py``). Built from the fractions
  rather than from ``cart_ang``, the distances do not depend on
  ``origin_ang``: from ``cart_ang``, an origin of 1e6 Å had moved distances
  by 1.4e-10 Å and per-atom sums by 3.6e-9 v.u.; now the same fractions in
  the same box give bit-identical results at any origin, and a Frame whose
  ``cart_ang`` sits up to its permitted 1e-9 Å from ``origin + frac @ box``
  is searched the same way by both methods.
* **The tree is asked a hair wider and the distances decide.** The cKDTree's
  own distance test can differ from the recomputed distance in the last bit,
  so the tree is queried at r + :data:`QUERY_PAD_ANG` and a pair is kept when
  the recomputed d <= r. The pair set is then exactly symmetric and the same
  for both search methods.

Because the two paths form a distance from different roundings, a contact's
valence here and in the crystal path can differ in its last bits (on the
eleven test crystals, 34-72 % of the listed contacts differ, by at most 51
units in the last place, 1.1e-14 v.u.). At a threshold away from every
valence that changes nothing. At a threshold set exactly equal to a
contact's valence, the strict ``v > v_bond`` can count that contact on one
path and not the other, and the plateau can move to its neighbour. A
threshold snapped to a step edge therefore has to be taken from the same
:class:`ValenceTable` (``v_vu``) it is applied to, never from a crystal-path
result.

THE TWO SEARCH METHODS
----------------------
``'images'`` starts from ``NeighborFinder``'s construction:
``_images_within(box_ang.T, r)`` gives at least 125 translations
(``neighbors.py:205`` adds 1 to ceil(r / width)), and every copy of every atom
goes into one tree. A copy whose fractional coordinate lies more than r / w
outside [0, 1) along an axis (w the perpendicular width) is farther than r
from the whole box, so it is left out (:func:`_image_points`): on the 29 791-
atom sheared box the tree held 3 723 875 points before and 50 653 now, with
the same pairs, and the image path went from 3.0 s to 1.6 s per frame
(pinned). ``'boxsize'`` is scipy's periodic
cKDTree, which applies the minimum-image convention [3, 4]. It returns only the
nearest image of each neighbour (measured in Step 0: with L = 5 Å and
r = 4.5 Å it returns the 1.0 Å pair and misses its 4.0 Å image), it refuses a
coordinate equal to L, and it accepts a negative box length without complaint.
So it is used only when every off-diagonal entry of the box is exactly 0,
every box length is positive, and r + :data:`QUERY_PAD_ANG` < min(L) / 2, where
each neighbour has one image within r; ``'auto'`` applies that rule. Both give
the same pair set, which a test checks.

Two independent implementations give the same pairs. Both were run in a
throwaway environment outside FACET (neither is a dependency, and no test
imports them), on the code as it stands. ASE 3.29.0's ``neighbor_list`` [5]
returned the same ordered pairs with the same lattice shifts as
:func:`find_pairs` on six frames: a 3 375-atom sheared box with a nonzero
origin, a 1 000-atom cubic box, 40 atoms in a strongly tilted cell with 9.0,
9.4 and 11.5 Å edges, where a neighbour has several images within the 7.5 Å
radius, 300 atoms in a left-handed general triclinic box, and the 608-atom
eulytite supercell on both paths. Distances and vectors agreed to 1.4e-14 Å,
and both counted the same 18 and 2 pairs below 0.4 Å. OVITO 3.16.1's
``CutoffNeighborFinder`` [6] returned the same pairs and the same periodic
shifts on four more: a 1 000-atom sheared box with a nonzero origin at
r = 7.5 Å (165 332 pairs), an 8-atom thin tilted cell where one pair has up to
19 images within r, a 1 331-atom cubic box on the periodic tree and a
343-atom left-handed triclinic box, to 1.8e-14 Å in distance and 2.0e-14 Å in
vector.

MEMORY
------
The search runs over blocks of centre atoms (:func:`iter_pairs`), and the
valence table keeps only the listed contacts of each block. The blocks are
cut from a coarse map of the local number density (:func:`_block_bounds`),
so that each holds about :data:`PAIRS_PER_BLOCK` ordered pairs for a slab
with vacuum or a cluster in a large box as well as for a uniform glass: on
29 791 atoms the largest block held 0.51-0.54 million pairs in all four
geometries of the table below, where blocks sized from the average density
had held 1.6 million (slab) and 2.2 million (cluster). The pair arrays
therefore stay bounded whatever the frame size and shape. What grows with N
is the padded table, (N, K) with K the longest contact list, at 1.0-1.2 KB
per atom for K = 12-14 (97-114 MB at 103 823 atoms), with its temporaries
while it is built; and the search structure, which is small beside it (the
periodic tree holds N points, the image tree 1.4-2.1 N at r = 6 Å, 70 bytes
a point).

TIMINGS AND MEMORY
------------------
Measured 2026-10-06 on Windows 11, Intel i5-13420H, Python 3.11.9, numpy
2.4.6, scipy 1.15.1, on a machine shared with other sessions (Step 0 found
single runs there vary by about 30 %). The boxes are the Step 0 benchmark's
(``tools/bench_md.make_box``: a jittered 2.3 Å grid, Si/O/O/Na drawn at
random; the chemistry is meaningless, only the size matters): cubic; sheared
to xy, xz, yz = 0.30, 0.20, 0.25 L; a slab, the same atoms in a box three
times as tall (two thirds vacuum); and a cluster, the same cube in the
middle of a box five times as wide. ``bv.DEFAULT``, default thresholds,
search radius 6.0 Å (the floor of ``search_radius_for``). ``analyse_frame``
end to end, median [min-max] of 5 runs after one untimed run (3 runs at
103 823 atoms), one configuration per process, once pinned to the P-cores
and once unpinned, as an ordinary script runs. The extra memory is the peak
private commit of the process above its commit with the frame built,
``scipy.spatial`` imported and the bond-valence table loaded, pinned runs
(the unpinned ones were within 2 MB of them); those one-off costs are left
out (importing ``scipy.spatial`` alone raised the commit by 372 MB, once per
process)::

      atoms  box      method      pairs  pinned (s)        unpinned (s)       MB
      9 261  cubic    boxsize   740 880  0.51 [0.48-0.58]   0.95 [0.55-1.13]   78
      9 261  cubic    images *  740 880  0.51 [0.44-0.65]   0.55 [0.39-1.02]   80
      9 261  sheared  images    736 878  0.53 [0.48-0.74]   0.59 [0.37-0.96]   91
     10 648  cubic    boxsize   851 840  0.63 [0.53-0.67]   1.10 [0.73-1.31]   94
     10 648  cubic    images *  851 840  0.58 [0.56-0.64]   0.89 [0.44-1.16]   99
     10 648  sheared  images    846 960  0.61 [0.55-0.67]   0.81 [0.47-1.28]   99
     29 791  cubic    boxsize  2 383 280 1.96 [1.90-2.03]   3.46 [3.38-3.54]  153
     29 791  cubic    images * 2 383 280 1.72 [1.55-1.79]   3.02 [2.88-3.32]  160
     29 791  sheared  images   2 369 864 1.56 [1.53-1.64]   3.57 [3.17-4.17]  161
     29 791  slab     boxsize  2 308 322 1.57 [1.55-1.78]   3.19 [2.62-3.61]  150
     29 791  cluster  boxsize  2 164 674 1.59 [1.51-1.69]   2.91 [2.86-2.98]  137
    103 823  cubic    boxsize  8 305 840 5.64 [5.63-5.78]  11.77 [11.53-12.42] 281
    103 823  sheared  images   8 259 440 6.51 [6.33-6.72]  12.72 [12.50-13.43] 269

    pairs: ordered pairs; MB: the extra memory
    * method='images' forced on a box where 'auto' takes 'boxsize'

The search alone (the blocks drained, no table) was about 60-80 % of each
total. Moving the threshold (:func:`at_threshold`) took 4-10 ms at 9-11 k
atoms, 15-17 ms at 30 k and 53-54 ms at 104 k pinned, about twice that
unpinned. The frames were cut into 2 blocks at 9-11 k atoms, 5 at 30 k and
17 at 104 k, the largest holding 0.50-0.59 million pairs. Before the image
tree was cut down to the copies within reach of the box, the same sheared
boxes took 1.17 s (9 261 atoms) and 3.03 s (29 791) pinned, with 155 and
302 MB, and a 97 336-atom sheared box 13.2 s and 692 MB; the image path is
now as fast as the periodic tree. Step 0 measured
``coordination.analyse_structure`` at 26 s on a 9 261-atom cubic box of the
same recipe, pinned. Reading a file is not included
(``md_model.frame_from_arrays`` took 4-7 ms at 9-11 k atoms).

REFERENCES
----------
[1] N. E. Brese and M. O'Keeffe, "Bond-valence parameters for solids",
    *Acta Crystallographica* B47 (1991) 192-197.
[2] I. D. Brown, "Recent developments in the methods and applications of the
    bond valence model", *Chemical Reviews* 109 (2009) 6858-6919.
[3] M. P. Allen and D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed.,
    Oxford University Press, Oxford (2017).
[4] P. Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific
    computing in Python", *Nature Methods* 17 (2020) 261-272.
[5] A. H. Larsen et al., "The atomic simulation environment - a Python
    library for working with atoms", *Journal of Physics: Condensed Matter*
    29 (2017) 273002.
[6] A. Stukowski, "Visualization and analysis of atomistic simulation data
    with OVITO - the Open Visualization Tool", *Modelling and Simulation in
    Materials Science and Engineering* 18 (2010) 015012.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np

from . import bv
from .md_model import Frame, _ox_label, symbols_changed_by_normalise

__all__ = [
    "D_MIN_ANG", "RADIUS_FLOOR_ANG", "RADIUS_CEILING_ANG", "PAIRS_PER_BLOCK",
    "QUERY_PAD_ANG", "IMAGE_POINTS_LIMIT", "DENSITY_CELLS_MAX", "OX_ABS_LIMIT",
    "METHODS",
    "search_radius_ang", "PairTable", "iter_pairs", "find_pairs",
    "ParamTable", "parameter_table", "ValenceTable", "valence_table",
    "AtomResults", "at_threshold", "analyse_frame", "Bonds", "bonds_at",
    "distance_cn",
]

# ---------------------------------------------------------------------------
# constants: the first three are NeighborFinder's and search_radius_for's own
# defaults, pinned to them by tests/test_bulk.py; the rest are choices
# ---------------------------------------------------------------------------

# Contacts at or below this distance are left out, as NeighborFinder(dmin=0.4)
# does (neighbors.py:89-90, 128-131): "a contact below about 0.4 A is the atom
# found in its own image". Here they are also counted.
D_MIN_ANG: float = 0.4

# search_radius_for's floor and ceiling (neighbors.py:221, 237).
RADIUS_FLOOR_ANG: float = 6.0
RADIUS_CEILING_ANG: float = 12.0

# About how many ordered pairs one block of centre atoms holds. A memory
# choice, not a physical value: a block's arrays peaked at 220-250 bytes per
# pair (private commit, measured 2026-10-06), so 500 000 pairs is about
# 110-125 MB at the peak of a block.
PAIRS_PER_BLOCK: int = 500_000

# How much wider than r the tree is queried before the recomputed distance
# decides. A numerical choice: it only has to exceed the difference between
# the tree's distance and the recomputed one, which are formed from the same
# fractions and box (Frame.cart_ang is never read) and differ by rounding:
# at most 3.6e-14 Å over 646 000 pairs in a 71 Å box, measured.
QUERY_PAD_ANG: float = 1e-9

# The most image points the 'images' search puts in its tree. A memory choice,
# not a physical value: a point held 70 bytes once the tree was built and
# 149 bytes at the peak of building it (measured), so the limit is about
# 0.7 GB held and 1.5 GB at the peak. A model reaches it only with a radius
# several times the box width (each atom then has about (1 + 2 r / w)^3
# copies; a 103 823-atom box at r = 6 Å has 148 877 points); it is refused
# with the estimate rather than ending in a MemoryError inside numpy.
IMAGE_POINTS_LIMIT: int = 10_000_000

# At most this many cells per box axis in the number-density map that sizes
# the blocks. A cost choice: the map only decides where one block ends and the
# next begins, never enters a result.
DENSITY_CELLS_MAX: int = 64

# The largest |oxidation state| accepted. A parsing bound, not a chemical one:
# every whole number up to it is exact as a float and as an int64, so the
# cast keeps each state and its sign (without it, a state of 1e20 casts to
# -2**63 and a cation becomes an anion).
OX_ABS_LIMIT: int = 2 ** 31 - 1

METHODS = ("auto", "images", "boxsize")


# ---------------------------------------------------------------------------
# argument checks shared by every entry point
# ---------------------------------------------------------------------------

def _check_frame(frame) -> Frame:
    if not isinstance(frame, Frame):
        raise ValueError(f"a Frame is needed, not {type(frame).__name__}; "
                         "md_model.frame_from_arrays builds one")
    return frame


def _positive_float(value, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} {value!r} is not a number")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} {value!r} is not a number") from None
    if not math.isfinite(out) or out < 0.0 or (out == 0.0 and not allow_zero):
        raise ValueError(f"{name} is {value!r}; a finite number "
                         f"{'of 0 or more' if allow_zero else 'above 0'} "
                         "is needed")
    return out


def _threshold(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} {value!r} is not a number")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} {value!r} is not a number") from None
    if not math.isfinite(out):
        raise ValueError(f"{name} is {value!r}; a finite valence is needed")
    return out


def _ox_array(ox_atom, elements: np.ndarray) -> np.ndarray:
    """(N,) int64 oxidation states, or ValueError naming the atoms without one.

    Accepted: integers, whole-number floats. An atom whose state is None or
    NaN has no state, and is refused rather than classed by
    ``elements.is_anion_like``.
    """
    n = int(np.asarray(elements).shape[0])
    raw = np.asarray(ox_atom)
    if raw.shape != (n,):
        raise ValueError(f"ox_atom has shape {raw.shape}; one oxidation state "
                         f"per atom ({n},) is needed")
    if raw.dtype.kind == "b":
        raise ValueError("ox_atom holds booleans, not oxidation states")
    if raw.dtype.kind in "iu":
        _ox_in_range(raw.astype(np.float64))
        return raw.astype(np.int64)
    if raw.dtype.kind == "O":
        missing = np.array([x is None for x in raw], dtype=bool)
        if missing.any():
            rows = np.flatnonzero(missing)
            raise ValueError(
                f"{rows.size} atom(s) have no oxidation state (first rows "
                f"{rows[:5].tolist()}, elements "
                f"{sorted({str(elements[r]) for r in rows[:50]})}); every atom "
                "needs one (md_model.model_oxidation gives a state per element)")
        try:
            raw = raw.astype(np.float64)
        except (TypeError, ValueError):
            raise ValueError("ox_atom holds values that are not numbers") \
                from None
    if raw.dtype.kind != "f":
        raise ValueError(f"ox_atom of type {raw.dtype} is not numeric")
    bad = ~np.isfinite(raw)
    if bad.any():
        rows = np.flatnonzero(bad)
        raise ValueError(
            f"{rows.size} atom(s) have no oxidation state (NaN; first rows "
            f"{rows[:5].tolist()}); every atom needs one")
    if (raw != np.round(raw)).any():
        raise ValueError("ox_atom holds states that are not whole numbers")
    _ox_in_range(raw)
    return raw.astype(np.int64)


def _ox_in_range(states: np.ndarray) -> None:
    """ValueError naming the rows whose |state| exceeds OX_ABS_LIMIT."""
    outside = np.abs(states) > OX_ABS_LIMIT
    if outside.any():
        rows = np.flatnonzero(outside)
        raise ValueError(
            f"{rows.size} atom(s) have an oxidation state beyond "
            f"+-{OX_ABS_LIMIT} (OX_ABS_LIMIT, a parsing bound; first rows "
            f"{rows[:5].tolist()}, values {states[rows[:5]].tolist()}); no "
            "atom carries such a charge")


def _frozen(array: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(array)
    array.setflags(write=False)
    return array


# ---------------------------------------------------------------------------
# the search radius
# ---------------------------------------------------------------------------

def _species(elements: np.ndarray, ox: np.ndarray):
    """Cation (element, state) species and anion elements, alphabetical."""
    cation_mask = ox >= 0
    cations = sorted({(str(e), int(o)) for e, o in
                      zip(elements[cation_mask], ox[cation_mask])})
    anions = sorted({str(e) for e in elements[~cation_mask]})
    return cations, anions


def search_radius_ang(elements, ox_atom, params: bv.ParameterSet,
                      v_list_vu: float, *, floor_ang: float = RADIUS_FLOOR_ANG,
                      ceiling_ang: float = RADIUS_CEILING_ANG) -> float:
    """The radius that finds every contact above ``v_list_vu``.

    ``neighbors.search_radius_for``'s rule (``neighbors.py:220-237``): the
    largest ``BVParam.distance_for(v_list)`` over the cation-anion
    combinations present, kept between ``floor_ang`` and ``ceiling_ang``. Taken
    over every (element, state) cation species, not one state per element as
    its dict at ``:229-230`` keeps, so it equals ``search_radius_for`` when
    each element has one state and is larger, never smaller, otherwise.
    Elements that ``elements.normalise`` turns into another element (He, Kr,
    ...) are left out, as their contacts carry no parameter here.
    """
    symbols = np.asarray(elements)
    ox = _ox_array(ox_atom, symbols)
    v_list = _positive_float(v_list_vu, "v_list_vu")
    floor = _positive_float(floor_ang, "floor_ang", allow_zero=True)
    ceiling = _positive_float(ceiling_ang, "ceiling_ang")
    if ceiling < floor:
        raise ValueError(f"ceiling_ang {ceiling} is below floor_ang {floor}")
    renamed = symbols_changed_by_normalise({str(s) for s in symbols})
    cations, anions = _species(symbols, ox)
    want = floor
    for cation, state in cations:
        if cation in renamed:
            continue
        for anion in anions:
            if anion in renamed:
                continue
            p = params.get(cation, state, anion)
            if p is not None:
                want = max(want, p.distance_for(v_list))
    return float(min(max(want, floor), ceiling))


# ---------------------------------------------------------------------------
# the pair search
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class PairTable:
    """Ordered periodic pairs within ``r_ang`` of a block of centre atoms.

    Every pair whose centre ``i`` lies in ``centre_start .. centre_stop - 1``
    is here, from that centre's end; the same pair seen from ``j`` is in the
    block that holds ``j``. ``image`` is the lattice translation applied to
    ``j``: the neighbour sits at ``cart_ang[j] + image @ box_ang``.
    ``vec_ang`` is centre -> neighbour image; rows of pairs beyond
    ``vectors_within_ang`` hold NaN, and it is None when no vectors were asked
    for (a distance-only pass). ``frame_digest`` identifies the fractions and
    box the pairs were searched on, so that :func:`valence_table` and
    :func:`distance_cn` refuse the pairs of another frame with the same
    number of atoms (the next frame of a trajectory, say).
    """

    i: np.ndarray                  # (P,) int32 centre atom (row)
    j: np.ndarray                  # (P,) int32 neighbour atom (row)
    image: np.ndarray              # (P, 3) int16
    d_ang: np.ndarray              # (P,)
    vec_ang: np.ndarray | None     # (P, 3) or None
    r_ang: float
    d_min_ang: float
    n_below_d_min: int             # non-self ordered pairs at d <= d_min_ang
    method: str                    # 'images' | 'boxsize'
    n_atoms: int = 0
    centre_start: int = 0
    centre_stop: int = 0
    vectors_within_ang: float | None = None
    closest_below_d_min: tuple[int, int, float] | None = None   # rows, d
    frame_digest: str = ""

    def __len__(self) -> int:
        return int(self.d_ang.shape[0])


def _frame_digest(frame: Frame) -> str:
    """BLAKE2b of what the pair search reads: the fractions and the box.

    Not the elements or the origin, which do not move a pair: the same
    positions with other elements, or shifted as a whole, have the same pairs.
    Measured at 0.6-2.4 ms for 9 261-29 791 atoms and 7 ms for 103 823,
    pinned to the P-cores.
    """
    digest = hashlib.blake2b(digest_size=16)
    digest.update(repr(frame.frac.shape).encode())
    digest.update(np.ascontiguousarray(frame.frac).tobytes())
    digest.update(np.ascontiguousarray(frame.box_ang).tobytes())
    return digest.hexdigest()


class _Coverage:
    """Checks the pair blocks a consumer is given, one block at a time.

    Every block has to come from a search on this frame, with one radius and
    one d_min, and between them the blocks have to cover every centre atom
    exactly once. A generator from :func:`iter_pairs` yields its blocks once,
    so a second consumer of the same generator receives none; that, a block
    given twice and a block left out are all refused here, rather than counted
    as atoms without neighbours.
    """

    def __init__(self, frame: Frame | None, n_atoms: int):
        self.digest = None if frame is None else _frame_digest(frame)
        self.n_atoms = n_atoms
        self.count = np.zeros(n_atoms, dtype=np.int64)
        self.first: PairTable | None = None

    def add(self, block: PairTable) -> None:
        if block.n_atoms != self.n_atoms:
            raise ValueError(f"the pair table is for {block.n_atoms} atoms and "
                             f"the frame holds {self.n_atoms}")
        if self.digest is not None and block.frame_digest != self.digest:
            raise ValueError(
                "the pairs were searched on another frame (its fractional "
                "positions or box differ from this frame's); iter_pairs on "
                "this frame gives its own")
        if self.first is None:
            self.first = block
        elif block.r_ang != self.first.r_ang or \
                block.d_min_ang != self.first.d_min_ang:
            raise ValueError("the pair blocks were searched with different "
                             "radii or d_min")
        self.count[block.centre_start:block.centre_stop] += 1

    def check(self) -> None:
        if self.first is None:
            raise ValueError(
                "no pair table was given (a generator from iter_pairs yields "
                "its blocks once; a second consumer needs a list of the blocks "
                "or a second search)")
        if (self.count != 1).any():
            twice = int((self.count > 1).sum())
            never = int((self.count == 0).sum())
            raise ValueError(f"the pair blocks cover {never} centre atom(s) not "
                             f"at all and {twice} more than once; each atom "
                             "needs exactly one block")


def _boxsize_refusal(frame: Frame, r_ang: float) -> str | None:
    """Why the periodic cKDTree cannot be used here, or None when it can."""
    if not frame.box_is_diagonal:
        return ("the box has a nonzero off-diagonal entry (a tilted box); the "
                "periodic cKDTree takes three edge lengths only")
    lengths = np.diag(frame.box_ang)
    if (lengths <= 0.0).any():
        return ("a box length on the diagonal is not positive; the periodic "
                "cKDTree accepts that and returns distances in another box")
    if not r_ang + QUERY_PAD_ANG < float(lengths.min()) / 2.0:
        return (f"r = {r_ang:.6g} Å is not below half the smallest box length "
                f"({float(lengths.min()) / 2.0:.6g} Å), so a neighbour can have "
                "more than one image within r and the periodic cKDTree returns "
                "only the nearest")
    return None


def _checked_block_atoms(block_atoms) -> int | None:
    """None, or a whole number of 1 or more; ValueError otherwise."""
    if block_atoms is None:
        return None
    if isinstance(block_atoms, (bool, np.bool_)) or \
            not isinstance(block_atoms, (int, np.integer)) or block_atoms < 1:
        raise ValueError(f"block_atoms {block_atoms!r}: a whole number of 1 "
                         "or more is needed")
    return int(block_atoms)


def _block_bounds(frame: Frame, r_ang: float, block_atoms
                  ) -> list[tuple[int, int]]:
    """(start, stop) row ranges of the centre blocks.

    ``block_atoms`` given: blocks of that many rows. Otherwise each atom's
    pair count is estimated from the number density of the cell of a coarse
    map it sits in (cells at least r wide, at most DENSITY_CELLS_MAX per
    axis), and a block ends where the running estimate passes another
    PAIRS_PER_BLOCK. The average density would do for a uniform glass, but a
    slab with vacuum or a cluster in a large box holds its atoms at several
    times the average, and a block sized from the average then held 3-4 times
    PAIRS_PER_BLOCK pairs (one block of 2.2 million pairs for a 29 791-atom
    cluster in a box five times its size).

    The block of each atom is numbered from its own running estimate, block
    m holding the atoms whose estimate lies in (m P, (m + 1) P] with P =
    PAIRS_PER_BLOCK, so the work is one pass over the atoms however large the
    estimate. Listing the multiples of P instead needs one entry per P
    estimated pairs: at r = 1e5 Å in an 11.5 Å box of 30 atoms that was 5e9
    entries, and the process grew past 17 GB before the image-count refusal
    was reached (measured 2026-10-06).
    """
    n = frame.n_atoms
    block_atoms = _checked_block_atoms(block_atoms)
    if block_atoms is not None:
        size = min(block_atoms, n)
        return [(s, min(n, s + size)) for s in range(0, n, size)]
    bins = np.clip(np.floor(frame.perpendicular_widths_ang / r_ang), 1,
                   DENSITY_CELLS_MAX).astype(np.int64)
    cell = np.minimum((frame.frac * bins).astype(np.int64), bins - 1)
    flat = np.ravel_multi_index(tuple(cell.T), tuple(bins))
    count = np.bincount(flat, minlength=int(bins.prod()))
    sphere = 4.0 / 3.0 * math.pi * r_ang ** 3
    estimate = count[flat] * (float(bins.prod()) / frame.volume_ang3) * sphere
    running = np.cumsum(np.maximum(estimate, 1.0))
    block = np.ceil(running / PAIRS_PER_BLOCK)              # non-decreasing
    starts = np.flatnonzero(block[1:] != block[:-1]) + 1
    edges = np.concatenate(([0], starts, [n]))               # within [0, n]
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def _lattice(frac: np.ndarray, box: np.ndarray) -> np.ndarray:
    """f_a a + f_b b + f_c c, term by term, for (P, 3) fractions."""
    return frac[:, 0:1] * box[0] + frac[:, 1:2] * box[1] + frac[:, 2:3] * box[2]


def _pair_geometry(frac: np.ndarray, box: np.ndarray, i: np.ndarray,
                   j: np.ndarray, image: np.ndarray):
    """(vec, d): vec = (f_j - f_i + n) . (a, b, c), from the fractions.

    Formed from the fractions rather than from ``cart_ang``, so the origin
    never enters: ``cart_ang`` carries ``origin_ang``, and at an origin of
    1e6 Å its rounding had moved distances by 1.4e-10 Å and per-atom sums by
    3.6e-9 v.u. Written term by term rather than as a matrix product, so that
    the vector from j to the image -n of i is bit for bit the negative of
    this one: f_i - f_j is exactly -(f_j - f_i), adding -n to it gives
    exactly -(f_j - f_i + n), and each product and sum changes only its sign.
    """
    df = (frac[j] - frac[i]) + image.astype(np.float64)
    vec = _lattice(df, box)
    d = np.linalg.norm(vec, axis=1)               # as neighbors.py:126
    return vec, d


def _image_extent(frame: Frame, r_ang: float):
    """(lo, count): per atom and axis, the first image index within reach of
    the box and how many there are (:func:`_image_points` explains the reach).

    One pass over the atoms, nothing the size of the tree is built, so
    :func:`iter_pairs` runs it before anything else. ValueError when an image
    index would leave the int16 range the tables store (a box thinner than
    r / 32 766, 2e-4 Å at r = 6 Å, or a radius over 32 766 box widths), and
    when the tree would exceed IMAGE_POINTS_LIMIT.
    """
    frac = frame.frac
    widths = frame.perpendicular_widths_ang
    margin = (r_ang + QUERY_PAD_ANG) / widths
    reach = np.ceil(margin) + 1.0
    top = float(np.iinfo(np.int16).max)
    if (reach > top).any():
        k = int(np.argmax(reach))
        raise ValueError(
            f"the box is {widths[k]:.4g} Å wide across its {'abc'[k]} faces, so "
            f"r = {r_ang:.6g} Å reaches images {reach[k]:.6g} boxes away, "
            f"beyond the +-{int(top)} the int16 image indices hold; a box that "
            "thin, or a radius that large beside the box, is refused")
    lo = np.ceil(-margin - frac).astype(np.int64)          # (N, 3), <= 0
    hi = np.floor(1.0 + margin - frac).astype(np.int64)    # (N, 3), >= 0
    count = hi - lo + 1
    total_estimate = float(np.prod(count.astype(np.float64), axis=1).sum())
    if total_estimate > IMAGE_POINTS_LIMIT:
        raise ValueError(
            f"the image search would hold {total_estimate:.3g} points "
            f"({frame.n_atoms} atoms, r = {r_ang:.6g} Å, box widths "
            f"{', '.join(f'{w:.4g}' for w in widths)} Å), above "
            f"IMAGE_POINTS_LIMIT ({IMAGE_POINTS_LIMIT}, a memory choice); a "
            "radius closer to the box width needs fewer images")
    return lo, count


def _image_points(frame: Frame, r_ang: float):
    """(atom, image, position) of every image point within reach of the box.

    ``NeighborFinder`` puts all (2 reps + 1)^3 translated copies of every atom
    in its tree, at least 125 N points (``neighbors.py:99-110``). Most cannot
    pair with any atom of the home cell: the distance from a point to the
    face plane s_k = 0 of the box is |s_k| w_k, with s its fractional
    coordinate and w_k the perpendicular width, so a copy with s_k below
    -r / w_k or above 1 + r / w_k is farther than r from every point of the
    home cell. Only the copies inside that margin (plus QUERY_PAD_ANG) are
    kept, which leaves the pair set unchanged and cut the tree of a 29 791-atom
    sheared box at r = 6 Å from 3 723 875 points to 50 653.

    ValueError from :func:`_image_extent` when the image indices or the tree
    would not fit.
    """
    frac, box = frame.frac, frame.box_ang
    lo, count = _image_extent(frame, r_ang)
    per_atom = np.prod(count, axis=1)
    total = int(per_atom.sum())
    atom = np.repeat(np.arange(frame.n_atoms, dtype=np.int64), per_atom)
    offset = np.arange(total, dtype=np.int64) - np.repeat(
        np.cumsum(per_atom) - per_atom, per_atom)
    image = np.empty((total, 3), dtype=np.int64)
    for axis in (2, 1, 0):                     # mixed radix, c fastest
        size = count[atom, axis]
        image[:, axis] = lo[atom, axis] + offset % size
        offset //= size
    del offset
    position = _lattice(frac[atom] + image, box)
    return atom, image.astype(np.int16), position


def iter_pairs(frame: Frame, r_ang: float, *, d_min_ang: float = D_MIN_ANG,
               block_atoms: int | None = None,
               vectors_within_ang: float | None = None,
               method: str = "auto") -> Iterator[PairTable]:
    """The one pair search per frame, one block of centre atoms at a time.

    ``'images'``: the translated copies ``NeighborFinder`` builds
    (``neighbors.py:99-110``), less those farther than r from every point of
    the box (:func:`_image_points`), in one cKDTree. ``'boxsize'``: scipy's
    periodic cKDTree, allowed only when every off-diagonal box entry is
    exactly 0, every length is positive and r_ang + QUERY_PAD_ANG < min(L) / 2;
    ``ValueError`` otherwise. ``'auto'`` takes ``'boxsize'`` when that rule
    allows it and ``'images'`` when it does not. Both search, and measure,
    the origin-free positions ``frac @ box_ang``; ``cart_ang`` is not read.

    Pairs at d <= ``d_min_ang`` are left out, as ``NeighborFinder`` does, and
    the non-self ones counted in ``n_below_d_min``. ``block_atoms=None`` sizes
    the blocks from a number-density map so each holds about
    :data:`PAIRS_PER_BLOCK` pairs (:func:`_block_bounds`).
    ``vectors_within_ang``: None keeps every pair's vector; a value keeps
    those with d <= value and stores NaN in the others; 0 stores none.
    ValueError for a box so thin, or a radius so large, that the image indices
    or the image tree would not fit (:func:`_image_extent`).

    Every argument is checked, and those refusals raised, when ``iter_pairs``
    is called, so a refusal names the call that caused it; the trees are
    built when the first block is asked for.
    """
    frame = _check_frame(frame)
    r = _positive_float(r_ang, "r_ang")
    d_min = _positive_float(d_min_ang, "d_min_ang", allow_zero=True)
    if d_min >= r:
        raise ValueError(f"d_min_ang {d_min} is not below r_ang {r}")
    if vectors_within_ang is not None:
        vectors_within_ang = _positive_float(vectors_within_ang,
                                             "vectors_within_ang",
                                             allow_zero=True)
    if method not in METHODS:
        raise ValueError(f"method {method!r}: one of {METHODS}")
    block_atoms = _checked_block_atoms(block_atoms)
    refusal = _boxsize_refusal(frame, r)
    if method == "boxsize" and refusal is not None:
        raise ValueError("method 'boxsize' cannot be used: " + refusal)
    chosen = "boxsize" if (method == "boxsize" or
                           (method == "auto" and refusal is None)) else "images"
    if chosen == "images":
        # the image count before anything that grows with the radius: the
        # block map's estimate grows as r^3 too (_block_bounds)
        _image_extent(frame, r)
    return _search(frame, r, d_min, block_atoms, vectors_within_ang, chosen)


def _search(frame: Frame, r: float, d_min: float, block_atoms: int | None,
            vectors_within_ang: float | None, chosen: str
            ) -> Iterator[PairTable]:
    """The blocks of :func:`iter_pairs`, its arguments already checked."""
    from scipy.spatial import cKDTree

    n = frame.n_atoms
    box = frame.box_ang
    frac = frame.frac
    r_query = r + QUERY_PAD_ANG

    if chosen == "images":
        point_atom, point_image, points = _image_points(frame, r)
        tree = cKDTree(points)
        del points
        home = _lattice(frac, box)
    else:
        lengths = np.diag(box).copy()
        # frac @ box for a diagonal box, relative to the origin.
        # cKDTree(boxsize=) refuses a coordinate equal to L. Under IEEE
        # round-to-nearest, f * L stays below L for every f < 1: the exact
        # product is at least half an ulp of L below L, and the double nearest
        # to it is below L (a
        # scratch probe over 2 000 017 lengths and the 64 largest fractions
        # below 1 found no exception). The next line is therefore a guard that
        # is not expected to fire; a position it moves lies within one
        # rounding of the box face, and the recomputed distances below decide.
        relative = frac * lengths
        relative[relative >= lengths] = 0.0
        tree = cKDTree(relative, boxsize=lengths)
    bounds = _block_bounds(frame, r, block_atoms)
    digest = _frame_digest(frame)

    for start, stop in bounds:
        if chosen == "images":
            centres = cKDTree(home[start:stop])
            found = centres.sparse_distance_matrix(tree, r_query,
                                                   output_type="ndarray")
            i = found["i"] + start
            k = found["j"]
            j = point_atom[k]
            image = point_image[k]
        else:
            centres = cKDTree(relative[start:stop], boxsize=lengths)
            found = centres.sparse_distance_matrix(tree, r_query,
                                                   output_type="ndarray")
            i = found["i"] + start
            j = found["j"]
            # the minimum image, from the exact fractions rather than the
            # guarded positions; unique because r < min(L) / 2
            image = (-np.rint(frac[j] - frac[i])).astype(np.int16)
        del found
        vec, d = _pair_geometry(frac, box, i, j, image)
        below = d <= d_min
        own = (i == j) & ~image.any(axis=1)
        dropped = below & ~own
        n_below = int(dropped.sum())
        closest = None
        if n_below:
            w = np.flatnonzero(dropped)
            m = w[np.argmin(d[w])]
            closest = (int(i[m]), int(j[m]), float(d[m]))
        keep = ~below & (d <= r)
        i, j, image, d, vec = i[keep], j[keep], image[keep], d[keep], vec[keep]
        if vectors_within_ang is not None:
            if vectors_within_ang <= d_min:
                vec = None
            else:
                vec[d > vectors_within_ang] = np.nan
        yield PairTable(
            i=_frozen(i.astype(np.int32)), j=_frozen(j.astype(np.int32)),
            image=_frozen(image), d_ang=_frozen(d),
            vec_ang=None if vec is None else _frozen(vec),
            r_ang=r, d_min_ang=d_min, n_below_d_min=n_below, method=chosen,
            n_atoms=n, centre_start=start, centre_stop=stop,
            vectors_within_ang=vectors_within_ang,
            closest_below_d_min=closest, frame_digest=digest)


def find_pairs(frame: Frame, r_ang: float, **kw) -> PairTable:
    """:func:`iter_pairs` concatenated into one table over every centre."""
    blocks = list(iter_pairs(frame, r_ang, **kw))
    first = blocks[0]
    vec = None if first.vec_ang is None else \
        _frozen(np.concatenate([b.vec_ang for b in blocks]))
    closest = [b.closest_below_d_min for b in blocks
               if b.closest_below_d_min is not None]
    return PairTable(
        i=_frozen(np.concatenate([b.i for b in blocks])),
        j=_frozen(np.concatenate([b.j for b in blocks])),
        image=_frozen(np.concatenate([b.image for b in blocks])),
        d_ang=_frozen(np.concatenate([b.d_ang for b in blocks])),
        vec_ang=vec, r_ang=first.r_ang, d_min_ang=first.d_min_ang,
        n_below_d_min=sum(b.n_below_d_min for b in blocks),
        method=first.method, n_atoms=first.n_atoms, centre_start=0,
        centre_stop=first.n_atoms, vectors_within_ang=first.vectors_within_ang,
        closest_below_d_min=min(closest, key=lambda c: c[2]) if closest
        else None, frame_digest=first.frame_digest)


# ---------------------------------------------------------------------------
# the parameters
# ---------------------------------------------------------------------------

CATEGORIES = ("fitted", "estimated", "none")


@dataclass(frozen=True)
class ParamTable:
    """One ``ParameterSet.get`` per (cation element, state, anion element).

    ``category`` is ``'fitted'`` (``BVParam.fitted``: a published table or a
    user override), ``'estimated'`` (the O'Keeffe-Brese expression) or
    ``'none'`` (no parameter: those contacts carry no valence).
    """

    param: dict[tuple[str, int, str], bv.BVParam | None]
    category: dict[tuple[str, int, str], str]
    params_name: str
    notes: tuple[str, ...] = ()

    def label(self, key: tuple[str, int, str]) -> str:
        """``'Si4+-O'`` for ``('Si', 4, 'O')``."""
        cation, state, anion = key
        return f"{_ox_label(cation, state)}-{anion}"


def parameter_table(elements, ox_atom, params: bv.ParameterSet | None
                    ) -> ParamTable:
    """``params.get(cation, cation_ox, anion)`` once per combination present.

    The argument order ``analyse_site`` uses from a cation centre and from an
    anion centre alike (``coordination.py:255-261``). A combination with an
    element ``elements.normalise`` turns into another one (He -> H, Kr -> K,
    ...; ``md_model.symbols_changed_by_normalise``) is given no parameter,
    because ``ParameterSet.get`` would look up the other element's.
    """
    params = params or bv.DEFAULT
    symbols = np.asarray(elements)
    ox = _ox_array(ox_atom, symbols)
    renamed = symbols_changed_by_normalise({str(s) for s in symbols})
    cations, anions = _species(symbols, ox)
    param: dict = {}
    category: dict = {}
    notes: list[str] = []
    for cation, state in cations:
        for anion in anions:
            key = (cation, state, anion)
            if cation in renamed or anion in renamed:
                param[key], category[key] = None, "none"
                continue
            p = params.get(cation, state, anion)
            param[key] = p
            category[key] = ("none" if p is None else
                             "fitted" if p.fitted else "estimated")
    table = ParamTable(param, category, str(getattr(params, "name", "")))
    if renamed:
        notes.append(
            "no bond-valence parameter is looked up for " + ", ".join(
                f"{s} (elements.normalise reads it as {t})"
                for s, t in renamed.items())
            + "; their contacts carry no valence")
    estimated = [k for k, c in category.items() if c == "estimated"]
    if estimated:
        notes.append(
            "estimated, not fitted, bond-valence parameters for "
            + ", ".join(f"{table.label(k)} (R0 {param[k].r0:.4f} Å, "
                        f"{param[k].source})" for k in estimated))
    if not cations:
        notes.append("the model holds no cation (no atom with a state of 0 or "
                     "more), so no contact carries a bond valence")
    if not anions:
        notes.append("the model holds no anion (no atom with a negative "
                     "state), so no contact carries a bond valence")
    return ParamTable(param, category, table.params_name, tuple(notes))


# ---------------------------------------------------------------------------
# the valence table
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class ValenceTable:
    """Counter-ion contacts with v > v_list per atom, valence-descending.

    Row r holds atom r's listed contacts, padded to (N, K) with K the longest
    list: ``nbr`` -1, ``d_ang`` NaN, ``vec_ang`` and ``v_vu`` 0 in the padding.
    Ties in valence are ordered by neighbour row, then image, so the table does
    not depend on the search order. ``cum_v_vu[r, c]`` is the sum of the first
    c + 1 valences of row r, and ``cum_bvv_vu[r, c]`` the sum of v times the
    unit vector, so any threshold is a lookup (:func:`at_threshold`).

    ``n_unparameterised`` counts each atom's counter-ion contacts within
    ``r_search_ang`` that have no parameter; ``uses_estimated`` is True when
    any of them uses an estimated one (``coordination.py:253-268``). The
    per-pair dicts count each cation-anion contact once, from its cation end.
    """

    elements: np.ndarray           # (N,) the frame's symbols
    ox: np.ndarray                 # (N,) int64
    is_anion: np.ndarray           # (N,) bool: ox < 0, as Site.is_anion
    nbr: np.ndarray                # (N, K) int32, -1 pad
    image: np.ndarray              # (N, K, 3) int16
    d_ang: np.ndarray              # (N, K) NaN pad
    vec_ang: np.ndarray            # (N, K, 3) zero pad
    v_vu: np.ndarray               # (N, K) 0.0 pad
    n_listed: np.ndarray           # (N,) contacts with v > v_list_vu
    cum_v_vu: np.ndarray           # (N, K)
    cum_bvv_vu: np.ndarray         # (N, K, 3)
    n_unparameterised: np.ndarray  # (N,)
    uses_estimated: np.ndarray     # (N,) bool
    missing_pairs: dict[str, int]  # 'Ti4+-O' -> contacts without a valence
    estimated_pairs: dict[str, int]
    params: ParamTable
    v_list_vu: float
    r_search_ang: float
    n_below_d_min: int
    method: str
    notes: tuple[str, ...] = ()

    @property
    def n_atoms(self) -> int:
        return int(self.nbr.shape[0])


def _blocks_of(pairs) -> Iterator[PairTable]:
    if isinstance(pairs, PairTable):
        yield pairs
        return
    for block in pairs:
        if not isinstance(block, PairTable):
            raise ValueError(f"a PairTable or an iterable of them is needed, "
                             f"not {type(block).__name__}")
        yield block


def _lookup_arrays(symbols: np.ndarray, ox: np.ndarray, table: ParamTable):
    """Per-atom codes into the parameter grid, and the grid of groups.

    ``group[c, a]`` is the index of the parameter for cation species c and
    anion element a in ``groups`` (a list of (key, BVParam | None)).
    """
    cations, anions = _species(symbols, ox)
    cation_code = {k: n for n, k in enumerate(cations)}
    anion_code = {k: n for n, k in enumerate(anions)}
    code = np.full(symbols.shape[0], -1, dtype=np.int64)
    tokens, inverse = np.unique(symbols, return_inverse=True)
    inverse = inverse.reshape(-1)
    for t, token in enumerate(tokens):
        rows = inverse == t
        neg = rows & (ox < 0)
        if neg.any():
            code[neg] = anion_code[str(token)]
        for state in np.unique(ox[rows & (ox >= 0)]):
            code[rows & (ox == state)] = cation_code[(str(token), int(state))]
    groups = []
    grid = np.zeros((max(1, len(cations)), max(1, len(anions))), dtype=np.int64)
    for c, (cation, state) in enumerate(cations):
        for a, anion in enumerate(anions):
            key = (cation, state, anion)
            grid[c, a] = len(groups)
            groups.append((key, table.param[key]))
    return code, grid, groups


def valence_table(frame: Frame, pairs, ox_atom, params: bv.ParameterSet | None
                  = None, *, v_list_vu: float = bv.V_LIST_DEFAULT,
                  r_search_ang: float | None = None) -> ValenceTable:
    """The listed counter-ion contacts of every atom, from one pair search.

    ``pairs`` is a :class:`PairTable` or an iterable of the blocks
    :func:`iter_pairs` yields; between them they must cover every centre atom
    exactly once, and they must come from a search on this frame's fractions
    and box (``PairTable.frame_digest``). Counter-ions are split by the sign of the oxidation state
    (never ``elements.is_anion_like``; an atom with no state is refused).
    ``BVParam.valence`` is called once per parameter on its distance array.
    ``r_search_ang`` (default ``pairs.r_ang``) limits the contacts the flags
    and counts look at, so a search run wider for a g(r) gives the same table
    as one at the bond-valence radius; it may not exceed the search radius.
    A radius shorter than the distance at which some parameter's valence
    falls to ``v_list_vu`` leaves listed contacts out, and a note says so
    with both distances.
    """
    frame = _check_frame(frame)
    params = params or bv.DEFAULT
    symbols = frame.elements
    n = frame.n_atoms
    ox = _ox_array(ox_atom, symbols)
    v_list = _positive_float(v_list_vu, "v_list_vu")
    is_anion = ox < 0
    table = parameter_table(symbols, ox, params)
    code, grid, groups = _lookup_arrays(symbols, ox, table)

    coverage = _Coverage(frame, n)
    n_unparam = np.zeros(n, dtype=np.int64)
    uses_est = np.zeros(n, dtype=bool)
    missing: dict[str, int] = {}
    estimated: dict[str, int] = {}
    kept = {"i": [], "j": [], "image": [], "d": [], "vec": [], "v": []}
    n_below = 0
    closest = None
    r_used = None
    d_min = None
    method = None

    for block in _blocks_of(pairs):
        coverage.add(block)
        if r_used is None:
            r_pairs = block.r_ang
            r_used = r_pairs if r_search_ang is None else \
                _positive_float(r_search_ang, "r_search_ang")
            if r_used > r_pairs:
                raise ValueError(
                    f"r_search_ang {r_used} exceeds the radius the pairs were "
                    f"searched to ({r_pairs} Å); contacts beyond it are missing")
            d_min, method = block.d_min_ang, block.method
        n_below += block.n_below_d_min
        if block.closest_below_d_min is not None and (
                closest is None or block.closest_below_d_min[2] < closest[2]):
            closest = block.closest_below_d_min

        i, j, d = block.i, block.j, block.d_ang
        select = (is_anion[i] != is_anion[j]) & (d <= r_used)
        if not select.any():
            continue
        if block.vec_ang is None:
            raise ValueError("the pairs carry no vectors (vectors_within_ang "
                             "was 0); the valence table needs them")
        i, j, d = i[select], j[select], d[select]
        vec, image = block.vec_ang[select], block.image[select]
        if np.isnan(vec).any():
            raise ValueError(
                f"the pairs carry vectors only within "
                f"{block.vectors_within_ang} Å; the valence table needs them "
                f"out to {r_used} Å")
        centre_anion = is_anion[i]
        cation_row = np.where(centre_anion, j, i)
        anion_row = np.where(centre_anion, i, j)
        group = grid[code[cation_row], code[anion_row]]
        v = np.full(d.shape[0], np.nan)
        for g in np.unique(group):
            key, p = groups[g]
            sel = group == g
            label = table.label(key)
            from_cation = int((sel & ~centre_anion).sum())
            if p is None:
                n_unparam += np.bincount(i[sel], minlength=n)
                missing[label] = missing.get(label, 0) + from_cation
                continue
            v[sel] = p.valence(d[sel])                    # bv.py:67-69
            if not p.fitted:
                uses_est[i[sel]] = True
                estimated[label] = estimated.get(label, 0) + from_cation
        listed = v > v_list                               # NaN compares False
        for name, values in (("i", i), ("j", j), ("image", image), ("d", d),
                             ("vec", vec), ("v", v)):
            kept[name].append(values[listed])

    coverage.check()

    def joined(name, empty_shape, dtype):
        parts = kept[name]
        return np.concatenate(parts) if parts else np.zeros(empty_shape, dtype)

    i = joined("i", (0,), np.int32)
    j = joined("j", (0,), np.int32)
    image = joined("image", (0, 3), np.int16)
    d = joined("d", (0,), np.float64)
    vec = joined("vec", (0, 3), np.float64)
    v = joined("v", (0,), np.float64)
    order = np.lexsort((image[:, 2], image[:, 1], image[:, 0], j, -v, i))
    i, j, image, d, vec, v = (i[order], j[order], image[order], d[order],
                              vec[order], v[order])
    n_listed = np.bincount(i, minlength=n).astype(np.int64)
    width = int(n_listed.max()) if n else 0
    start = np.cumsum(n_listed) - n_listed
    column = np.arange(i.shape[0]) - start[i]

    nbr_pad = np.full((n, width), -1, dtype=np.int32)
    image_pad = np.zeros((n, width, 3), dtype=np.int16)
    d_pad = np.full((n, width), np.nan)
    vec_pad = np.zeros((n, width, 3))
    v_pad = np.zeros((n, width))
    nbr_pad[i, column] = j
    image_pad[i, column] = image
    d_pad[i, column] = d
    vec_pad[i, column] = vec
    v_pad[i, column] = v
    unit = np.zeros((n, width, 3))
    unit[i, column] = vec / d[:, None]               # as bv.phi_index, :547-549
    cum_v = np.cumsum(v_pad, axis=1)
    cum_bvv = np.cumsum(v_pad[:, :, None] * unit, axis=1)

    notes = list(table.notes)
    reach = [(p.distance_for(v_list), key) for key, p in table.param.items()
             if p is not None]
    if reach and max(reach)[0] > r_used:
        need, key = max(reach)
        where = ("search_radius_ang gives a radius that holds them"
                 if need <= RADIUS_CEILING_ANG else
                 f"that distance lies beyond the {RADIUS_CEILING_ANG:g} Å "
                 "ceiling of search_radius_ang, RADIUS_CEILING_ANG; an "
                 "r_search_ang that wide, given explicitly, holds them")
        notes.append(
            f"contacts were taken out to {r_used:.6g} Å, and {table.label(key)} "
            f"keeps a valence above {v_list:g} v.u. out to {need:.6g} Å; "
            f"contacts between the two distances are not in the table ({where})")
    if missing:
        notes.append(
            "no bond-valence parameter for " + ", ".join(
                f"{k} ({c} contact{'s' if c != 1 else ''})"
                for k, c in sorted(missing.items()))
            + f" within {r_used:.6g} Å; those contacts carry no valence, are in "
            "no sum, and are counted per atom in n_unparameterised")
    if n_below:
        a, b, dist = closest
        notes.append(
            f"{n_below} ordered pair(s) at or below {d_min} Å (the "
            "NeighborFinder dmin) are left out of every count; the closest is "
            f"{dist:.4f} Å between atom ids {int(frame.atom_id[a])} "
            f"({symbols[a]}) and {int(frame.atom_id[b])} ({symbols[b]})")
    zero = sorted({str(s) for s in symbols[ox == 0]})
    if zero:
        notes.append(
            f"{', '.join(zero)} with oxidation state 0 count on the cation side "
            "of the split, as Site.is_anion places a state of 0 "
            "(structure.py:59-63); their contacts to anions are looked up as "
            "(element, 0, anion)")

    return ValenceTable(
        elements=symbols, ox=_frozen(ox), is_anion=_frozen(is_anion),
        nbr=_frozen(nbr_pad), image=_frozen(image_pad), d_ang=_frozen(d_pad),
        vec_ang=_frozen(vec_pad), v_vu=_frozen(v_pad),
        n_listed=_frozen(n_listed), cum_v_vu=_frozen(cum_v),
        cum_bvv_vu=_frozen(cum_bvv), n_unparameterised=_frozen(n_unparam),
        uses_estimated=_frozen(uses_est), missing_pairs=dict(sorted(
            missing.items())), estimated_pairs=dict(sorted(estimated.items())),
        params=table, v_list_vu=v_list, r_search_ang=float(r_used),
        n_below_d_min=n_below, method=method, notes=tuple(notes))


# ---------------------------------------------------------------------------
# one threshold
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class AtomResults:
    """Every atom at one v_bond: the SiteResult quantities, field for field.

    NaN where the SiteResult field is NaN: the sums, |BVV| and phi of an atom
    with no contact in the set, and the plateau width off a plateau.
    ``bvv_vector_vu`` is zero where CN is 0, as ``SiteResult.bvv_vector``.
    """

    v_bond_vu: float
    v_list_vu: float
    cn: np.ndarray                 # (N,) v > v_bond_vu     = SiteResult.cn_valence
    cn_listed: np.ndarray          # (N,) v > v_list_vu     = cn_listed
    bvs_vu: np.ndarray             #                        = bvs
    bvs_listed_vu: np.ndarray      #                        = bvs_listed
    bvv_vu: np.ndarray             # |BVV|                  = bvv
    bvv_listed_vu: np.ndarray      #                        = bvv_listed
    bvv_vector_vu: np.ndarray      # (N, 3)                 = bvv_vector
    phi: np.ndarray                #                        = phi
    phi_listed: np.ndarray         #                        = phi_listed
    plateau_decades: np.ndarray    #                        = plateau_decades
    valence_discrepancy_vu: np.ndarray   # bvs - |ox|       = valence_discrepancy
    notes: tuple[str, ...] = ()


def at_threshold(table: ValenceTable,
                 v_bond_vu: float = bv.V_BOND_DEFAULT) -> AtomResults:
    """Every field at ``v_bond_vu``, from the table: no search, no exp.

    cn = #(v > v_bond); the sums, the bond-valence vector and phi are the
    running sums at column cn - 1; the plateau is k = #(v >= v_bond), width
    log10(v_k / v_(k+1)) with v_list below the last contact, NaN when k = 0
    or when k is the whole list and v_bond <= v_list (the module docstring
    gives the crystal lines this mirrors). Every field is recomputed at every
    threshold, the listed ones included.
    """
    if not isinstance(table, ValenceTable):
        raise ValueError(f"a ValenceTable is needed, not {type(table).__name__}")
    v_bond = _threshold(v_bond_vu, "v_bond_vu")
    v = table.v_vu
    n, width = v.shape
    rows = np.arange(n)
    valid = np.arange(width)[None, :] < table.n_listed[:, None]

    def at_column(count, cum):
        """cum[r, count - 1], NaN (or 0 for vectors) where count is 0."""
        has = count > 0
        index = np.where(has, count - 1, 0)
        if cum.ndim == 3:
            out = np.zeros((n, 3))
            if width:
                out[has] = cum[rows, index][has]
            return out
        out = np.full(n, np.nan)
        if width:
            out[has] = cum[rows, index][has]
        return out

    cn = ((v > v_bond) & valid).sum(axis=1).astype(np.int64)
    listed = table.n_listed.astype(np.int64)
    bvs = at_column(cn, table.cum_v_vu)
    bvs_listed = at_column(listed, table.cum_v_vu)
    vector = at_column(cn, table.cum_bvv_vu)
    vector_listed = at_column(listed, table.cum_bvv_vu)
    has, has_listed = cn > 0, listed > 0
    bvv = np.where(has, np.linalg.norm(vector, axis=1), np.nan)
    bvv_listed = np.where(has_listed, np.linalg.norm(vector_listed, axis=1),
                          np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        phi = np.where(has, bvv / bvs, np.nan)
        phi_listed = np.where(has_listed, bvv_listed / bvs_listed, np.nan)

    # the plateau: Plateau.contains is v_low < v <= v_high (coordination.py:81-82)
    k = ((v >= v_bond) & valid).sum(axis=1)
    plateau = np.full(n, np.nan)
    on = (k > 0) & ~((k == listed) & (v_bond <= table.v_list_vu))
    if width and on.any():
        v_high = v[rows, np.where(k > 0, k - 1, 0)]
        inner = k < listed
        v_low = np.where(inner, v[rows, np.where(inner, k, 0)],
                         table.v_list_vu)
        with np.errstate(invalid="ignore", divide="ignore"):
            widths = np.where((v_low > 0) & (v_high > 0),
                              np.log10(v_high / v_low), np.inf)
        plateau[on] = widths[on]

    discrepancy = np.where(has, bvs - np.abs(table.ox), np.nan)

    notes = []
    empty = cn == 0
    if empty.any():
        symbols, counts = np.unique(table.elements[empty], return_counts=True)
        notes.append(
            f"{int(empty.sum())} atom(s) have no contact above {v_bond:g} v.u.: "
            + ", ".join(f"{s} {c}" for s, c in zip(symbols, counts)))

    def frozen(*arrays):
        return tuple(_frozen(np.asarray(a)) for a in arrays)

    (cn, listed, bvs, bvs_listed, bvv, bvv_listed, vector, phi, phi_listed,
     plateau, discrepancy) = frozen(cn, listed, bvs, bvs_listed, bvv,
                                    bvv_listed, vector, phi, phi_listed,
                                    plateau, discrepancy)
    return AtomResults(
        v_bond_vu=v_bond, v_list_vu=table.v_list_vu, cn=cn, cn_listed=listed,
        bvs_vu=bvs, bvs_listed_vu=bvs_listed, bvv_vu=bvv,
        bvv_listed_vu=bvv_listed, bvv_vector_vu=vector, phi=phi,
        phi_listed=phi_listed, plateau_decades=plateau,
        valence_discrepancy_vu=discrepancy, notes=tuple(notes))


def analyse_frame(frame: Frame, ox_atom, params: bv.ParameterSet | None = None,
                  *, v_bond_vu: float = bv.V_BOND_DEFAULT,
                  v_list_vu: float = bv.V_LIST_DEFAULT,
                  r_search_ang: float | None = None, method: str = "auto",
                  block_atoms: int | None = None
                  ) -> tuple[ValenceTable, AtomResults]:
    """search_radius_ang -> iter_pairs -> valence_table -> at_threshold.

    The pairs are consumed block by block and never all held at once. Keep the
    table to move the threshold afterwards with :func:`at_threshold`.
    """
    frame = _check_frame(frame)
    params = params or bv.DEFAULT
    ox = _ox_array(ox_atom, frame.elements)
    if r_search_ang is None:
        r_search_ang = search_radius_ang(frame.elements, ox, params, v_list_vu)
    r = _positive_float(r_search_ang, "r_search_ang")
    table = valence_table(
        frame, iter_pairs(frame, r, method=method, block_atoms=block_atoms),
        ox, params, v_list_vu=v_list_vu, r_search_ang=r)
    return table, at_threshold(table, v_bond_vu)


# ---------------------------------------------------------------------------
# bonds, and the distance-cut coordination number
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Bonds:
    """Cation-anion bonds at one v_bond (v > v_bond_vu), each bond once.

    Taken from the cation's row, so the anion sits at
    ``cart_ang[anion] + image @ box_ang`` and ``vec_ang`` points from the
    cation to it. The same bonds, seen from the anions, are those the anions'
    CN counts.
    """

    cation: np.ndarray             # (n_bonds,) row
    anion: np.ndarray              # (n_bonds,) row
    image: np.ndarray              # (n_bonds, 3) int16
    vec_ang: np.ndarray            # (n_bonds, 3) cation -> anion
    d_ang: np.ndarray              # (n_bonds,)
    v_vu: np.ndarray               # (n_bonds,)
    v_bond_vu: float

    def __len__(self) -> int:
        return int(self.cation.shape[0])


def bonds_at(table: ValenceTable, v_bond_vu: float = bv.V_BOND_DEFAULT
             ) -> Bonds:
    """The bonds the CN at ``v_bond_vu`` counts, each once.

    The single bond definition the glass descriptors use (bridging oxygens,
    Q^n, bond angles), so that they and the CN can never disagree.
    """
    if not isinstance(table, ValenceTable):
        raise ValueError(f"a ValenceTable is needed, not {type(table).__name__}")
    v_bond = _threshold(v_bond_vu, "v_bond_vu")
    v = table.v_vu
    n, width = v.shape
    valid = np.arange(width)[None, :] < table.n_listed[:, None]
    take = valid & (v > v_bond) & ~table.is_anion[:, None]
    rows, cols = np.nonzero(take)
    return Bonds(
        cation=_frozen(rows.astype(np.int32)),
        anion=_frozen(table.nbr[rows, cols]),
        image=_frozen(table.image[rows, cols]),
        vec_ang=_frozen(table.vec_ang[rows, cols]),
        d_ang=_frozen(table.d_ang[rows, cols]),
        v_vu=_frozen(v[rows, cols]), v_bond_vu=v_bond)


def distance_cn(pairs, elements, cutoff_ang: Mapping[tuple[str, str], float]
                ) -> dict[tuple[str, str], np.ndarray]:
    """Neighbours of element B within a distance cutoff of each atom of A.

    ``cutoff_ang`` maps (centre element, neighbour element) to a cutoff in Å;
    a neighbour counts when d <= cutoff, inclusive as ``SiteResult.cn_within``
    (``coordination.py:206-207``), and pairs at d <= d_min are not contacts, as
    there. The cutoffs are the caller's, typically the first minimum of the
    partial g(r); none is set here. Each value is an (N,) count; rows whose
    element is not the centre element hold -1, so a sum that takes in rows of
    another element comes out visibly short. ``elements`` is the frame's
    symbols, or the :class:`Frame` itself, and then the pairs are also
    checked to be that frame's. ValueError for a cutoff that is not positive
    or exceeds the radius the pairs were searched to, and, as in
    :func:`valence_table`, for blocks that do not cover every centre exactly
    once (a generator already consumed by another call yields none).
    """
    frame = elements if isinstance(elements, Frame) else None
    symbols = frame.elements if frame is not None else np.asarray(elements)
    n = symbols.shape[0]
    checked: dict[tuple[str, str], float] = {}
    for key, value in cutoff_ang.items():
        if not (isinstance(key, tuple) and len(key) == 2):
            raise ValueError(f"cutoff key {key!r}: a (centre element, "
                             "neighbour element) pair is needed")
        checked[(str(key[0]), str(key[1]))] = _positive_float(
            value, f"the cutoff for {key[0]}-{key[1]}")
    out = {key: np.where(symbols == key[0], 0, -1).astype(np.int64)
           for key in checked}
    coverage = _Coverage(frame, n)
    for block in _blocks_of(pairs):
        coverage.add(block)
        for key, cutoff in checked.items():
            if cutoff > block.r_ang:
                raise ValueError(
                    f"the cutoff for {key[0]}-{key[1]} is {cutoff} Å, beyond "
                    f"the {block.r_ang} Å the pairs were searched to")
            sel = (symbols[block.i] == key[0]) & (symbols[block.j] == key[1]) \
                & (block.d_ang <= cutoff)
            out[key] += np.bincount(block.i[sel], minlength=n)
    coverage.check()
    return {key: _frozen(value) for key, value in out.items()}
