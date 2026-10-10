"""Where mobile ions can move in an MD glass model: channels, three ways.

A modifier cation (Na, Li, K, Ag, Ca, ...) moves through a glass by jumping
from one site to another, and the sites it can reach form regions of the
model: channels when they run through the box, pockets when they do not.
Nothing in a frame labels those regions, so they are measured in three
complementary ways, each on one :class:`~facet.core.md_model.Frame`, each
giving a field or a label the 3D view can highlight and a few numbers that
compare between models:

* **by charge** (:func:`bv_landscape` and what follows it): the bond-valence
  landscape of a probe ion M^n+ on a periodic grid, its mismatch |V - n|,
  the volume fraction a mismatch threshold admits, the connected regions
  that fraction forms, and the smallest threshold at which a region runs
  through the box along a, b and c;
* **by modifier density** (:func:`modifier_density`): the number of
  modifier atoms within the M-O cutoff of every anion, the anions that hold
  k or more of them, and the clusters those anions and their modifiers form
  through M-O contacts;
* **by voids** (:func:`void_regions`): the empty spheres of the Delaunay
  tetrahedra (:func:`facet.core.md_order.empty_spheres`) grouped where they
  overlap into void regions, each with its volume, shape and the atoms that
  line it.

Every threshold, cutoff, spacing and probe radius is an argument: nothing is
set here, and every result states the values it was measured with.

BY CHARGE: THE BOND-VALENCE LANDSCAPE
-------------------------------------
Adams and Swenson [1] read the transport pathways of a mobile ion M in a
structure model from the bond-valence sum a probe M would have at every
point of the cell: where the sum V(x) over the anions lies close to the
formal valence n of M, the probe is bonded as the ion is at its own sites,
and the connected region where the mismatch |V(x) - n| stays below a
threshold is the pathway. Their observation, that the ionic conductivity of
a glass or a crystal follows the volume fraction such a region fills, is
what the accessible-volume curve here measures. Adams [2, 3] and Adams and
Rao [4] scale the mismatch to an energy and add a repulsion term for the
immobile cations; Chen, Wong and Adams [5] describe the softBV program that
implements that form and reports, among other things, the threshold at
which the pathways first percolate.

What is computed here, exactly:

* on the grid point x_ijk = (i/n_a) a + (j/n_b) b + (k/n_c) c of a grid with
  n_k = ceil(|a_k| / h) points along each cell vector (h the spacing asked
  for; the point of index 0 is the box corner, as :class:`facet.core.volume.Grid`
  lays a field out, so the field exports to it unchanged),
  ``V(x) = sum over anion images within r_cut of exp((R0 - d) / b)``, with
  one (R0, b) per anion element from ``params.get(probe, probe_ox, anion)``
  (:class:`facet.core.bv.ParameterSet`, FACET's Brese-O'Keeffe set unless
  another is given) and ``BVParam.valence`` evaluating the exponential, so a
  grid point that coincides with an atom of the probe element gives the sum
  ``bulk.valence_table`` lists for that atom (``tests/test_md_channels.py``
  checks it to 1e-10). An anion is an atom whose oxidation state is
  negative, the split the bulk engine uses; an anion element with no
  parameter contributes nothing and is named in the notes. The mismatch
  itself is kept, in valence units: no energy scaling is applied, so no
  barrier in eV can be read from it;
* ``r_cut_ang`` is required. The valence a contact carries at r_cut is
  stated per parameter, so the person who chose it can see what is left out
  (Na-O at 6 Å with R0 = 1.80 Å, b = 0.37 Å carries 1.2e-5 v.u.);
* an optional :class:`Repulsion`: grid points within ``r_excl_ang`` of a
  cation of the named elements (every cation element but the probe's, when
  none is named) are excluded from every accessible set. It stands in for
  the Coulomb repulsion term of the softBV form [3, 4] as a stated radius,
  not an energy. The probe's own atoms do not exclude anything, so the
  landscape shows where the mobile species could be, not where it is;
* the accessible fraction at a threshold Delta is the number of
  non-excluded grid points with mismatch <= Delta over all grid points, a
  volume fraction of the box;
* regions are the connected sets of accessible points under 6-connectivity
  (a point and its six neighbours along +-a, +-b, +-c, the structure
  ``scipy.ndimage.label`` uses by default [6]) with periodic boundaries: the
  labels of the non-periodic box are joined across each pair of opposite
  faces, and the translations a region maps onto itself by are found as
  :func:`facet.core.md_network.components` finds them for an atom graph
  (the lift of a periodic graph [7]: a spanning tree places every label in
  a cell, and each edge closing a cycle across a face contributes the net
  translation of that cycle). The rank of those translations is the
  region's dimensionality (0 a pocket, 1 a channel, 2 a sheet, 3 a
  framework) and a region spans axis a when one of them has a nonzero a
  component; ``copies`` is the index of that lattice, the number of disjoint
  pieces one label of the box stands for, as there;
* the percolation threshold along each axis is the smallest Delta at which
  some region spans it. Spanning is monotone in Delta (a larger threshold
  only adds points), so a bisection over the sorted mismatch values of the
  grid finds the exact grid value, in about log2(points) labellings per
  axis, shared between the axes; the accessible fraction at each threshold
  is reported with it. No tolerance parameter: the answer is a value of the
  grid. NaN when no path spans the axis even with every non-excluded point
  accessible (the exclusion zones close it);
* per region: the volume (points x the voxel volume V / (n_a n_b n_c)), the
  centroid of the points placed in one connected copy, the elongation,
  the lowest mismatch inside, the dimensionality, the spans and copies, and
  the atoms of the probe element whose nearest grid point lies in it. Every
  atom gets the region of its nearest grid point (``region_of_atom``, -1
  when that point is not accessible), for the 3D view.

**Elongation** is sqrt(lambda_max / lambda_min) of the gyration tensor of
the region, the tensor Theodorou and Suter [8] take the shape measures of a
chain from: for a grid region the covariance of the point positions plus
the covariance of one voxel, (1/12) sum_k (a_k / n_k)(a_k / n_k)^T, so a
single voxel has the shape of the voxel (1 in a cubic grid) and nothing
divides by zero; for a void region, the volume-weighted covariance of the
sphere centres plus each sphere's own term r^2 / 5 per axis, so one sphere
has elongation exactly 1. A finite region (dimensionality 0) is unwrapped
into one connected copy, which is unique. A region that spans the box has
no unique copy: where its points sit along the translation T that maps it
onto itself depends on where the box cuts it, and so would any tensor of
one copy. For a channel (dimensionality 1) the tensor is therefore the
spread of the points perpendicular to T, which every unwrapping shares,
plus |T|^2 / 12 along T, the spread of one period of a uniform line, so a
straight channel has lambda_max = |T|^2 / 12 against its cross-section,
well above 1, whatever the box cut. A sheet or a framework (dimensionality
2 or 3) has no elongation (NaN; the dimensionality says what it is), and
the centroid of any spanning region is that of its points as wrapped into
the box, which the notes state.

BY MODIFIER DENSITY: THE MODIFIED RANDOM NETWORK
------------------------------------------------
Greaves [9] read the EXAFS of alkali silicate glasses as a modified random
network: the modifier cations are not dispersed through the silicate
network but gathered in regions rich in modifiers and non-bridging oxygens,
which at a modifier content of a few tens of mol% join into percolating
channels, and Greaves and Ngai [10] built the ionic transport of oxide
glasses on those channels. Molecular dynamics models show the same
structure: Jund, Kob and Jullien [11] and Horbach, Kob and Binder [12]
found sodium diffusing along channels in sodium silicate models, and Meyer
et al. [13] traced the prepeak of the structure factor of sodium silicate
melts and glasses to the sodium-rich channels. Sami's own reading, that a
channel is where O atoms hold more Na than the average, is the one
measured here:

* for every anion of the elements named, the number of modifier atoms
  (``modifiers``, a required set) within the cutoff of their element pair;
  the cutoffs are the caller's (``cutoffs_ang`` with ``cutoff_sources``, as
  :func:`facet.core.md_network.graph_from_pairs` takes them), or
  :func:`measure_cutoffs` reads them from the first minimum of the partial
  g(r) of the frame through :func:`facet.core.glass.first_minimum` with a
  stated :class:`facet.core.glass.MinimumMethod`; a pair with no cutoff is
  refused, never guessed;
* an anion is modifier-rich when its count is k or more (``k_rich``,
  required). The distribution of the count over the anions is reported per
  frame and averaged over frames (:func:`average_modifier_density`);
* the clusters: the graph whose nodes are the modifier-rich anions and
  the modifier atoms within the cutoff of at least one of them, with an edge
  per such M-O contact, and its periodic connected components
  (:func:`facet.core.md_network.components`): sizes, the largest piece's
  share of the nodes and of the rich anions, dimensionality, which axes
  are spanned;
* when a :class:`facet.core.bulk.Bonds` object and the former set are given,
  the speciation of each anion (free, NBO, BO, tricluster by its bonded
  formers, the categories of :mod:`facet.core.glass`) is crossed with
  richness: of the modifier-rich anions, the fraction in each category,
  and of each category, the fraction that is modifier-rich.

BY VOIDS: OVERLAPPING EMPTY SPHERES
-----------------------------------
:func:`facet.core.md_order.empty_spheres` gives, for every Delaunay
tetrahedron of the periodic frame, the largest sphere centred on its
circumcentre that overlaps no atom sphere, the void description of Sastry
et al. [14]. Two neighbouring voids communicate where their spheres
overlap; here the spheres are grouped by that relation:

* spheres with ``radius_ang >= probe_radius_ang`` (and above 0) are the
  voids a probe of that radius fits in; the others are left out and
  counted. Spheres whose centres coincide to within
  ``md_order.DEGENERATE_REL_TOL`` of the mean atomic spacing are one void
  (atoms on one sphere give several tetrahedra one circumsphere, as in a
  lattice), and the copy with the largest radius is kept;
* two voids are connected when their probe-centre spheres, of radius
  r - probe_radius_ang, overlap: d < (r_i - r_p) + (r_j - r_p). The
  probe's centre can then pass from one sphere into the other inside
  their union. With a probe radius of 0 this is the overlap of the spheres
  themselves. Pairs are found with periodic images, and the regions are
  the periodic connected components of that graph, with their
  dimensionality, spans and copies as above;
* per region: the number of spheres, the sum of their volumes (overlaps
  counted as many times as they occur), the volume of their union
  measured on a periodic grid of spacing ``grid_spacing_ang`` (a grid point
  belongs to the sphere it lies deepest inside, so no point is counted
  twice and a point inside spheres of two regions, which the probe-shrunk
  criterion allows, goes to the deeper one), the volume-weighted centroid
  of the unwrapped centres, the elongation (above), the extent along the
  principal axis (from the farthest sphere surface to the farthest on the
  other side), the largest sphere radius, and the atoms lining it: an
  atom lines a region when the gap between its own sphere (the radius the
  spheres were measured with) and some sphere of the region is at most
  ``lining_distance_ang``; with 0, the atoms the spheres touch. Every atom
  gets the region whose sphere it is nearest to among those it lines
  (``region_of_atom``);
* :func:`select_regions` picks the regions with ``volume_ang3 >=
  min_volume_ang3`` and ``elongation >= min_elongation``, Sami's two
  criteria, on either kind of region.

INVARIANCES
-----------
Every quantity is periodic, and none depends on the order of the atoms (the
rows of a frame): the tests permute the rows of a 3000-atom glass and find
the same numbers to 1e-10. Translation: everything measured from the atoms
alone (per-atom bond-valence sums, modifier counts, clusters, the sphere
graph, its volumes, centroids relative to the atoms, elongations) is
invariant under any rigid translation of the model. A quantity sampled on
the grid (the landscape, its accessible fraction and regions, the union
volume of a void region) is invariant under translations by whole grid
steps, p a / n_a + q b / n_b + r c / n_c, which move the model and the grid
together; a translation by a fraction of a step resamples the field, and
the grid quantities then change by the order of a voxel at each region
boundary. The grid is anchored to the box corner rather than to an atom so
that the field exports to :class:`facet.core.volume.Grid` as it is.

NOT DONE HERE
-------------
No energy, barrier or conductivity is computed from the landscape. No
threshold, cutoff, radius or spacing has a default. The grid quantities are
not corrected for their voxel discretisation beyond what the notes state.
Nothing here runs or alters a simulation.

TIMINGS
-------
Measured 2026-10-09 on this machine (Windows 11, Intel i5-13420H, Python
3.11.9, numpy 2.4.6, scipy 1.15.1), unpinned, one run of each call in one
process per frame, with a test run of this module sharing the machine for
part of the larger frame (single runs here vary by about 30 %, bulk.py's
table). Wall time, the process CPU time in brackets where the two differ by
more than a tenth, and the peak private commit of the process above its
level before the call (sampled every 20 ms). The first landscape call of a
process also pays the import of scipy.spatial (bulk.py measured 372 MB,
once per process); that call's +667 MB is left out below. Two frames: the
3000-atom Na2O-3SiO2 glass (34.6 Å cubic box, 1750 O) and a 12 811-atom
Ca-Na aluminosilicate frame (55.5 Å cubic box, 7247 O), both on a 0.3 Å
grid, probe Na+, FACET's default parameters::

                                             3000 atoms            12 811 atoms
    grid points at 0.3 Å                     116^3 = 1.56 M        186^3 = 6.43 M
    bv_landscape, r_cut 6 Å                  8.8 s     +290 MB     31.9 s [29]  +330 MB
    bv_landscape, r_cut 8 Å                  19.1 s    +290 MB     105 s [81]   +330 MB
    bv_landscape, r_cut 6 Å, Repulsion 1 Å   11.5 s    +300 MB     34.9 s       +340 MB
    accessible_fraction, 5 thresholds        0.04 s                0.09 s
    accessible_regions at Delta 0.3          0.20 s    +38 MB      0.20 s       +120 MB
    percolation_thresholds                   1.4 s, 32 labellings  3.4 s, 34 labellings
    md_order.empty_spheres, half vdW radii   0.55 s, 20 596 spheres   2.5 s, 89 981 spheres
    void_regions, probe 0.5 Å, lining 0.5 Å  3.2 s     +350 MB     5.3 s        +510 MB
    bulk.find_pairs to 4 Å                   0.05 s                0.28 s
    modifier_density, k = 3                  0.01 s                0.10 s

The landscape is the one step whose cost grows with the grid and the cutoff
together. Both glasses hold 0.042 anions per Å^3, so each grid point sees 38
anion images within 6 Å: 59 million (grid point, anion image) pairs on the
3000-atom frame and 245 million on the larger one, about 7-8 million pairs
a second through ``cKDTree.sparse_distance_matrix`` and the exponential;
r_cut 8 Å has 2.4 times the pairs. Memory stays bounded because the pairs
are formed :data:`PAIRS_PER_CHUNK` (4 million) at a time, about 150-200 MB
at the peak of a chunk, and what grows with the grid is the three fields,
8 + 8 + 1 bytes a point (109 MB at 6.43 M points): the peak above the
process was 290-340 MB at both grid sizes. One labelling of the 6.43 M-point
grid (``scipy.ndimage.label``, the face joins, the component search) took
about 0.1 s, so the 34 labellings of the bisection took 3.4 s. The void
grouping spends its time in three cKDTree passes (coincident centres, the
overlap pairs, the lining atoms) and in marking the union on the grid; its
+510 MB is the label and depth grids (4 + 8 bytes a point) with the pair
chunks. What the runs gave, as measured values and nothing more: on the
3000-atom frame the Na+ landscape first spans a at Delta 0.158 v.u. and b, c
at 0.160 v.u. (accessible fraction 0.048), on the 12 811-atom frame at
0.235 and 0.233 v.u. (0.051); at Delta 0.3 the largest region of each is a
framework (dimensionality 3) holding most of the accessible volume.

REFERENCES
----------
[1] St. Adams and J. Swenson, "Determining ionic conductivity from
    structural models of fast ionic conductors", *Physical Review Letters*
    84 (2000) 4144-4147, https://doi.org/10.1103/PhysRevLett.84.4144
[2] S. Adams, "Relationship between bond valence and bond softness of alkali
    halides and chalcogenides", *Acta Crystallographica* B57 (2001)
    278-287, https://doi.org/10.1107/S0108768101003068
[3] S. Adams, "From bond valence maps to energy landscapes for mobile ions
    in ion-conducting solids", *Solid State Ionics* 177 (2006) 1625-1630,
    https://doi.org/10.1016/j.ssi.2006.03.054
[4] S. Adams and R. Prasada Rao, "Transport pathways for mobile ions in
    disordered solids from the analysis of energy-scaled bond-valence
    mismatch landscapes", *Physical Chemistry Chemical Physics* 11 (2009)
    3210, https://doi.org/10.1039/b901753d
[5] H. Chen, L. L. Wong and S. Adams, "SoftBV - a software tool for
    screening the materials genome of inorganic fast ion conductors", *Acta
    Crystallographica* B75 (2019) 18-33,
    https://doi.org/10.1107/S2052520618015718
[6] P. Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific
    computing in Python", *Nature Methods* 17 (2020) 261-272,
    https://doi.org/10.1038/s41592-019-0686-2; ``scipy.ndimage.label``
    documentation, "structure ... defaults to one in which the neighbors
    are those that touch along a face" (read 2026-10-09). The labelling
    itself is the cluster multiple labelling of Hoshen and Kopelman [15]
    in scipy's implementation.
[7] S. J. Chung, Th. Hahn and W. E. Klee, "Nomenclature and generation of
    three-periodic nets: the vector method", *Acta Crystallographica* A40
    (1984) 42-50, https://doi.org/10.1107/S0108767384000088
[8] D. N. Theodorou and U. W. Suter, "Shape of unperturbed linear polymers:
    polypropylene", *Macromolecules* 18 (1985) 1206-1214,
    https://doi.org/10.1021/ma00148a028
[9] G. N. Greaves, "EXAFS and the structure of glass", *Journal of
    Non-Crystalline Solids* 71 (1985) 203-217,
    https://doi.org/10.1016/0022-3093(85)90289-3
[10] G. N. Greaves and K. L. Ngai, "Reconciling ionic-transport properties
    with atomic structure in oxide glasses", *Physical Review B* 52 (1995)
    6358-6380, https://doi.org/10.1103/PhysRevB.52.6358
[11] P. Jund, W. Kob and R. Jullien, "Channel diffusion of sodium in a
    silicate glass", *Physical Review B* 64 (2001) 134303,
    https://doi.org/10.1103/PhysRevB.64.134303
[12] J. Horbach, W. Kob and K. Binder, "Dynamics of sodium in sodium
    disilicate: channel relaxation and sodium diffusion", *Physical Review
    Letters* 88 (2002) 125502, https://doi.org/10.1103/PhysRevLett.88.125502
[13] A. Meyer, J. Horbach, W. Kob, F. Kargl and H. Schober, "Channel
    formation and intermediate range order in sodium silicate melts and
    glasses", *Physical Review Letters* 93 (2004) 027801,
    https://doi.org/10.1103/PhysRevLett.93.027801
[14] S. Sastry, D. S. Corti, P. G. Debenedetti and F. H. Stillinger,
    "Statistical geometry of particle packings. I. Algorithm for exact
    determination of connectivity, volume, and surface areas of void space
    in monodisperse and polydisperse sphere packings", *Physical Review E*
    56 (1997) 5524-5532, https://doi.org/10.1103/PhysRevE.56.5524
[15] J. Hoshen and R. Kopelman, "Percolation and cluster distribution. I.
    Cluster multiple labeling technique and critical concentration
    algorithm", *Physical Review B* 14 (1976) 3438-3445,
    https://doi.org/10.1103/PhysRevB.14.3438
"""
from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from . import bulk, bv, glass, md_order
from .md_model import Frame, _ox_label, validate_symbol
from .md_network import (Components, _copies, _lattice_basis, _make_graph,
                         _unique_edges, components, graph_from_pairs)
from .md_order import EmptySpheres
from .md_stats import Distribution, Histogram, Scalar, Series

__all__ = [
    "PAIRS_PER_CHUNK", "AXES",
    "Repulsion", "Landscape", "bv_landscape", "accessible_fraction",
    "AccessibleRegions", "accessible_regions",
    "PercolationThresholds", "percolation_thresholds",
    "landscape_grid", "regions_grid", "region_indicator_grid",
    "MeasuredCutoffs", "measure_cutoffs",
    "ModifierDensity", "modifier_density",
    "VoidRegions", "void_regions", "select_regions",
    "average_accessible_fraction", "average_percolation_thresholds",
    "average_modifier_density", "average_void_regions",
]

# ---------------------------------------------------------------------------
# constants: memory choices and labels, never physical values
# ---------------------------------------------------------------------------

# About how many (grid point, anion image) pairs one chunk of the landscape
# holds. A memory choice: the structured array scipy returns is 24 bytes a
# pair, and the valences formed from it another 8-16, so 4 million pairs is
# about 150-200 MB at the peak of a chunk (measured 2026-10-09 on the
# 3000-atom NS3 glass: 38 pairs a point at r_cut = 6 Å).
PAIRS_PER_CHUNK: int = 4_000_000

# (sphere, grid offset) tests per chunk when marking the union of the void
# spheres, and grid points per chunk when the region moments are summed.
# Memory choices, never in a result.
_SPHERE_WORK_PER_CHUNK: int = 2_000_000
_POINTS_PER_CHUNK: int = 1_000_000

# The box axes, in the order of the box rows and of every (3,) array here.
AXES = ("a", "b", "c")

# The three ndimage structuring neighbours per side: a point's six face
# neighbours along +-a, +-b, +-c (scipy.ndimage.label's default, [6]).
_CONNECTIVITY_TEXT = ("6-connectivity: a grid point is joined to its two "
                      "neighbours along each of a, b and c, with periodic "
                      "boundaries")


# ---------------------------------------------------------------------------
# argument checks
# ---------------------------------------------------------------------------

def _check_frame(frame) -> Frame:
    if not isinstance(frame, Frame):
        raise ValueError(f"a Frame is needed, not {type(frame).__name__}; "
                         "md_model.frame_from_arrays builds one")
    return frame


def _nonnegative(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name}: a number is needed, not {value!r}")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: a number is needed, not {value!r}") from None
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} = {value!r}: a finite number of 0 or more is "
                         "needed")
    return out


def _positive(value, name: str) -> float:
    out = _nonnegative(value, name)
    if out == 0.0:
        raise ValueError(f"{name} = {value!r}: a number above 0 is needed")
    return out


def _whole(value, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or \
            not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} is {value!r}; a whole number is needed")
    if value < minimum:
        raise ValueError(f"{name} is {value}; at least {minimum} is needed")
    return int(value)


def _symbols(values, what: str) -> frozenset[str]:
    """A set of validated element symbols; no default and no bare string."""
    if values is None:
        raise ValueError(f"{what} has no default: give the element symbols "
                         "explicitly")
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{what} needs a collection of element symbols such "
                         f"as {{'Na'}}, not the text {values!r}")
    try:
        items = list(values)
    except TypeError:
        raise ValueError(f"{what} needs a collection of element symbols") \
            from None
    out = frozenset(validate_symbol(str(item)) for item in items)
    if not out:
        raise ValueError(f"{what} is empty")
    return out


def _frozen(array) -> np.ndarray:
    out = np.ascontiguousarray(array)
    out.setflags(write=False)
    return out


def _results(results, kind, what: str) -> list:
    results = list(results)
    if not results or not all(isinstance(r, kind) for r in results):
        raise ValueError(f"{what}: one {kind.__name__} per frame is needed")
    return results


def _one_setting(results, what: str, **getters) -> None:
    """ValueError when the frames' results differ in a named method
    parameter: an average over frames needs one method."""
    for label, get in getters.items():
        seen = []
        for r in results:
            value = get(r)
            if value not in seen:
                seen.append(value)
        if len(seen) > 1:
            raise ValueError(f"{what}: the frames were measured with "
                             f"different {label} ({seen}); an average over "
                             "frames needs one")


# ---------------------------------------------------------------------------
# the grid, and periodic images of arbitrary points
# ---------------------------------------------------------------------------

def _widths(box: np.ndarray) -> np.ndarray:
    """The box widths perpendicular to the bc, ca and ab faces (Å), as
    ``Frame.perpendicular_widths_ang``."""
    a, b, c = box
    volume = abs(float(np.linalg.det(box)))
    return np.array([volume / np.linalg.norm(np.cross(b, c)),
                     volume / np.linalg.norm(np.cross(c, a)),
                     volume / np.linalg.norm(np.cross(a, b))])


def _grid_shape(box: np.ndarray, spacing_ang: float) -> np.ndarray:
    """ceil(|a_k| / h) points along each cell vector, at least 1."""
    lengths = np.linalg.norm(box, axis=1)
    return np.maximum(1, np.ceil(lengths / spacing_ang)).astype(np.int64)


def _grid_frac(shape: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Fractional coordinates i/n of the grid points with flat index
    start .. stop-1 (C order: the a index slowest)."""
    i, j, k = np.unravel_index(np.arange(start, stop), tuple(shape))
    return np.column_stack([i / shape[0], j / shape[1], k / shape[2]])


def _images_within(frac: np.ndarray, box: np.ndarray, margin_ang: float):
    """(index, image, position) of every periodic image of the points
    ``frac`` (rows, in [0, 1)) within ``margin_ang`` of the box: the rule of
    ``bulk._image_points`` (a copy whose fractional coordinate lies more
    than margin / width outside [0, 1) along an axis is farther than the
    margin from every point of the box), for points that are not atoms.
    Positions are origin-free, ``(frac + image) . box``."""
    n = frac.shape[0]
    widths = _widths(box)
    reach = (margin_ang + bulk.QUERY_PAD_ANG) / widths
    lo = np.ceil(-reach - frac).astype(np.int64)
    hi = np.floor(1.0 + reach - frac).astype(np.int64)
    count = hi - lo + 1
    per_point = np.prod(count, axis=1)
    total = int(per_point.sum())
    if total > bulk.IMAGE_POINTS_LIMIT:
        raise ValueError(
            f"the image search would hold {total} points ({n} centres, margin "
            f"{margin_ang:.6g} Å), above bulk.IMAGE_POINTS_LIMIT "
            f"({bulk.IMAGE_POINTS_LIMIT}, a memory choice)")
    index = np.repeat(np.arange(n, dtype=np.int64), per_point)
    offset = np.arange(total, dtype=np.int64) - np.repeat(
        np.cumsum(per_point) - per_point, per_point)
    image = np.empty((total, 3), dtype=np.int64)
    for axis in (2, 1, 0):
        size = count[index, axis]
        image[:, axis] = lo[index, axis] + offset % size
        offset //= size
    position = bulk._lattice(frac[index] + image, box)
    return index, image, position


def _voxel_covariance(box: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """(1/12) sum_k (a_k / n_k)(a_k / n_k)^T: the covariance of a uniform
    distribution over one voxel (a parallelepiped spanned by the three
    grid steps)."""
    steps = box / shape[:, None]
    return steps.T @ steps / 12.0


def _elongation(cov: np.ndarray) -> float:
    """sqrt(lambda_max / lambda_min) of a symmetric 3 x 3 tensor; NaN when
    the smallest eigenvalue is not above 0 or the tensor is not finite."""
    if not np.isfinite(cov).all():
        return math.nan
    values = np.linalg.eigvalsh(cov)
    if not values[0] > 0.0:
        return math.nan
    return float(math.sqrt(values[-1] / values[0]))


def _principal_axis(cov: np.ndarray) -> np.ndarray:
    """The unit eigenvector of the largest eigenvalue (NaN for a tensor
    that is not finite)."""
    if not np.isfinite(cov).all():
        return np.full(3, np.nan)
    values, vectors = np.linalg.eigh(cov)
    return vectors[:, -1]


def _spanning_tensor(cov: np.ndarray, basis, box: np.ndarray, own: np.ndarray):
    """The gyration tensor of a region from the covariance ``cov`` of one
    unwrapping of its points, its translation ``basis`` (in cells) and its
    own term ``own`` (module docstring): rank 0 keeps cov; rank 1 keeps the
    part of cov perpendicular to the translation T, which every unwrapping
    shares, and takes |T|^2 / 12 along it; rank 2 and 3 give a tensor of
    NaN. Returns (tensor, |T| or NaN)."""
    rank = len(basis)
    if rank == 0:
        return cov + own, math.nan
    if rank == 1:
        t = np.asarray(basis[0], dtype=np.float64) @ box
        length = float(np.linalg.norm(t))
        unit = t / length
        along = np.outer(unit, unit)
        q = np.eye(3) - along
        return q @ cov @ q + (length * length / 12.0) * along + own, length
    return np.full((3, 3), np.nan), math.nan


# ---------------------------------------------------------------------------
# periodic connected components of a labelled set
# ---------------------------------------------------------------------------

def _periodic_components(n: int, u: np.ndarray, v: np.ndarray,
                         shift: np.ndarray):
    """Connected components of n nodes joined by edges (u, v, shift), where
    the edge joins node u in cell (0, 0, 0) to node v in cell ``shift``.

    Returns ``(label, cell, bases)``: the component of each node (0 .. C-1
    in order of the lowest node), the cell that places each node in one
    connected copy of its component, and per component the Hermite basis of
    the translations that map it onto itself (:func:`md_network.components`
    explains the construction; this is the same algorithm on nodes that are
    not atoms). A node with no edge is a component of its own, with no
    translation, and costs no search.
    """
    label = np.arange(n, dtype=np.int64)
    cell = np.zeros((n, 3), dtype=np.int64)
    bases: list[list] = [[] for _ in range(n)]
    if u.size:
        u = np.asarray(u, dtype=np.int64)
        v = np.asarray(v, dtype=np.int64)
        shift = np.asarray(shift, dtype=np.int64).reshape(-1, 3)
        ends = np.concatenate([u, v])
        order = np.argsort(ends, kind="stable")
        other = np.concatenate([v, u])[order].tolist()
        step = [tuple(r) for r in np.concatenate([shift, -shift])[order].tolist()]
        bounds = np.searchsorted(ends[order], np.arange(n + 1))
        touched = np.unique(ends)
        seen = np.zeros(n, dtype=bool)
        for root in touched.tolist():
            if seen[root]:
                continue
            seen[root] = True
            cells = {root: (0, 0, 0)}
            stack = [root]
            members = [root]
            while stack:
                x = stack.pop()
                cx = cells[x]
                for k in range(bounds[x], bounds[x + 1]):
                    y = other[k]
                    if not seen[y]:
                        seen[y] = True
                        t = step[k]
                        cells[y] = (cx[0] + t[0], cx[1] + t[1], cx[2] + t[2])
                        stack.append(y)
                        members.append(y)
            rows = np.array(members, dtype=np.int64)
            label[rows] = int(rows.min())
            for y in members:
                cell[y] = cells[y]
        defect = cell[u] + shift - cell[v]
        nonzero = defect.any(axis=1)
        if nonzero.any():
            rows = np.unique(np.column_stack([label[u[nonzero]],
                                              defect[nonzero]]), axis=0)
            for comp in np.unique(rows[:, 0]).tolist():
                vecs = [tuple(r) for r in rows[rows[:, 0] == comp][:, 1:].tolist()]
                bases[comp] = _lattice_basis(vecs)
    # relabel 0 .. C-1 in order of the lowest node of each component
    roots, inverse = np.unique(label, return_inverse=True)
    return inverse.reshape(-1).astype(np.int64), cell, \
        [bases[r] for r in roots.tolist()]


def _component_geometry(bases: Sequence):
    """(dimensionality, spans, copies) arrays from the translation bases."""
    count = len(bases)
    dimensionality = np.array([len(b) for b in bases], dtype=np.int64)
    spans = np.array([[any(vec[a] != 0 for vec in b) for a in range(3)]
                      for b in bases], dtype=bool).reshape(count, 3)
    copies = np.array([_copies(b) for b in bases], dtype=np.int64)
    return dimensionality, spans, copies


def _spans_text(spans: np.ndarray) -> str:
    axes = [AXES[k] for k in range(3) if spans[k]]
    return ", ".join(axes) if axes else "none"


# ---------------------------------------------------------------------------
# A. by charge: the bond-valence landscape of a probe ion
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Repulsion:
    """The exclusion that stands in for cation-cation repulsion (module
    docstring): grid points within ``r_excl_ang`` of a cation of ``elements``
    are excluded from every accessible set. ``elements`` None means every
    cation element of the model but the probe's; the probe's own atoms never
    exclude anything."""

    r_excl_ang: float
    elements: frozenset[str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "r_excl_ang",
                           _positive(self.r_excl_ang, "r_excl_ang"))
        if self.elements is not None:
            object.__setattr__(self, "elements",
                               _symbols(self.elements, "Repulsion.elements"))


@dataclass(frozen=True, eq=False)
class Landscape:
    """The bond-valence sum and mismatch of a probe on the grid of one frame.

    ``bvs_vu`` and ``mismatch_vu`` are indexed ``[i, j, k]`` with i along a
    (the layout of :class:`facet.core.volume.Grid`); the point of index 0 is
    the box corner and the point (i, j, k) is at fractional coordinates
    (i / n_a, j / n_b, k / n_c). ``excluded`` marks the points a
    :class:`Repulsion` removed, or is None. ``parameters`` maps each anion
    element to its (R0, b, source); ``anions_without_parameter`` names the
    anion elements that contribute nothing. The frame's symbols and
    fractions are kept so that regions can name the atoms they hold.
    """

    probe: str
    probe_ox: int
    bvs_vu: np.ndarray
    mismatch_vu: np.ndarray
    excluded: np.ndarray | None
    box_ang: np.ndarray
    origin_ang: np.ndarray
    shape: tuple[int, int, int]
    grid_spacing_ang: float
    spacing_ang: tuple[float, float, float]
    voxel_volume_ang3: float
    r_cut_ang: float
    params_name: str
    parameters: Mapping[str, tuple[float, float, str]]
    anions_without_parameter: tuple[str, ...]
    repulsion: Repulsion | None
    repulsion_elements: tuple[str, ...]
    elements: np.ndarray
    frac: np.ndarray
    frame_digest: str
    notes: tuple[str, ...] = ()

    @property
    def n_points(self) -> int:
        return int(self.mismatch_vu.size)

    @property
    def n_excluded(self) -> int:
        return 0 if self.excluded is None else int(np.count_nonzero(self.excluded))

    @property
    def target_vu(self) -> float:
        """|probe_ox|, the sum the probe matches."""
        return float(abs(self.probe_ox))

    @property
    def probe_label(self) -> str:
        return _ox_label(self.probe, self.probe_ox)

    @property
    def method_parameters(self) -> dict:
        out = {"probe": self.probe_label,
               "grid spacing requested (Å)": self.grid_spacing_ang,
               "grid spacing (Å)": self.spacing_ang,
               "grid shape": self.shape,
               "r_cut (Å)": self.r_cut_ang,
               "bond-valence parameters": self.params_name,
               "connectivity": _CONNECTIVITY_TEXT}
        if self.repulsion is None:
            out["repulsion exclusion"] = "none"
        else:
            out["repulsion exclusion radius (Å)"] = self.repulsion.r_excl_ang
            out["repulsion elements"] = self.repulsion_elements
        return out

    def sorted_mismatch(self) -> np.ndarray:
        """The mismatch of every non-excluded grid point, ascending."""
        if self.excluded is None:
            return np.sort(self.mismatch_vu.ravel())
        return np.sort(self.mismatch_vu[~self.excluded])


def bv_landscape(frame: Frame, ox_atom, probe: str, probe_ox: int, *,
                 grid_spacing_ang: float, r_cut_ang: float,
                 params: bv.ParameterSet | None = None,
                 repulsion: Repulsion | None = None) -> Landscape:
    """The bond-valence sum a probe ``probe``^``probe_ox`` would have at every
    point of a periodic grid over the frame, and its mismatch |V - n|.

    ``ox_atom`` gives every atom's oxidation state (``ModelOxidation.per_atom``);
    the anions are the atoms with a negative state. ``grid_spacing_ang`` and
    ``r_cut_ang`` are required (module docstring: the grid, the sum, what
    r_cut leaves out). ``params`` is the bond-valence parameter set
    (``bv.DEFAULT`` when None), looked up as ``params.get(probe, probe_ox,
    anion)``. ``repulsion`` excludes the grid points near the other cations.
    The sum runs in chunks of about :data:`PAIRS_PER_CHUNK` (grid point,
    anion image) pairs, so the memory does not grow with the grid beyond the
    three fields themselves.
    """
    from scipy.spatial import cKDTree

    frame = _check_frame(frame)
    params = params or bv.DEFAULT
    symbols = frame.elements
    ox = bulk._ox_array(ox_atom, symbols)
    probe = validate_symbol(probe)
    if isinstance(probe_ox, (bool, np.bool_)) or \
            not isinstance(probe_ox, (int, np.integer)):
        raise ValueError(f"probe_ox {probe_ox!r} is not a whole number")
    probe_ox = int(probe_ox)
    if probe_ox == 0:
        raise ValueError("a probe with oxidation state 0 has no valence to "
                         "match; give the formal charge of the mobile ion")
    h = _positive(grid_spacing_ang, "grid_spacing_ang")
    r_cut = _positive(r_cut_ang, "r_cut_ang")
    if repulsion is not None and not isinstance(repulsion, Repulsion):
        raise ValueError(f"repulsion needs a Repulsion, not "
                         f"{type(repulsion).__name__}")
    box = frame.box_ang
    shape = _grid_shape(box, h)
    n_points = int(np.prod(shape))
    is_anion = ox < 0
    if not is_anion.any():
        raise ValueError("the model holds no anion (no atom with a negative "
                         "oxidation state), so the probe has nothing to bond "
                         "to")
    anion_elements = sorted({str(s) for s in symbols[is_anion]})
    label = _ox_label(probe, probe_ox)
    parameters: dict[str, bv.BVParam] = {}
    missing: list[str] = []
    for element in anion_elements:
        p = params.get(probe, probe_ox, element)
        if p is None:
            missing.append(element)
        else:
            parameters[element] = p
    if not parameters:
        raise ValueError(
            f"no bond-valence parameter for {label} with "
            f"{', '.join(anion_elements)} in {params.name}; the landscape "
            "would be 0 everywhere")

    # anion images within r_cut of the box, coded by element
    atom, _, position = bulk._image_points(frame, r_cut)
    with_param = is_anion & np.isin(symbols, list(parameters))
    use = with_param[atom]
    position = position[use]
    codes = {element: c for c, element in enumerate(parameters)}
    tokens, inverse = np.unique(symbols[atom[use]], return_inverse=True)
    token_code = np.array([codes[str(t)] for t in tokens], dtype=np.int64)
    img_code = token_code[inverse.reshape(-1)]
    tree = cKDTree(position)
    del position

    # the repulsion exclusion
    excluded = None
    tree_c = None
    rep_elements: tuple[str, ...] = ()
    rep_notes: list[str] = []
    if repulsion is not None:
        cation_elements = sorted({str(s) for s in symbols[~is_anion]})
        if repulsion.elements is None:
            rep_elements = tuple(e for e in cation_elements if e != probe)
        else:
            rep_elements = tuple(sorted(repulsion.elements))
            outside = [e for e in rep_elements if e not in cation_elements]
            if outside:
                raise ValueError(
                    f"Repulsion.elements names {', '.join(outside)}, which "
                    "has no atom with an oxidation state of 0 or more in this "
                    "model; the exclusion applies to cations")
        rows = ~is_anion & np.isin(symbols, list(rep_elements))
        excluded = np.zeros(n_points, dtype=bool)
        if rows.any():
            atom_c, _, pos_c = bulk._image_points(frame, repulsion.r_excl_ang)
            tree_c = cKDTree(pos_c[rows[atom_c]])
            del pos_c
            rep_notes.append(
                f"repulsion exclusion: grid points within "
                f"{repulsion.r_excl_ang:g} Å of a {', '.join(rep_elements)} "
                f"atom ({int(rows.sum())} atoms) are excluded from every "
                "accessible set; the probe's own atoms exclude nothing")
        else:
            rep_notes.append(
                f"repulsion exclusion asked for around {', '.join(rep_elements)}"
                ", of which the model holds no atom; no grid point is excluded")

    # the sum, chunk by chunk
    density = int(with_param.sum()) / frame.volume_ang3
    estimate = max(1.0, density * 4.0 / 3.0 * math.pi * r_cut ** 3)
    per_chunk = int(min(n_points, max(1, PAIRS_PER_CHUNK // estimate)))
    bvs = np.zeros(n_points)
    param_list = [(parameters[element], codes[element]) for element in parameters]
    for start in range(0, n_points, per_chunk):
        stop = min(n_points, start + per_chunk)
        points = bulk._lattice(_grid_frac(shape, start, stop), box)
        found = cKDTree(points).sparse_distance_matrix(
            tree, r_cut, output_type="ndarray")
        if found.shape[0]:
            i, j, d = found["i"], found["j"], found["v"]
            del found
            code = img_code[j]
            total = np.zeros(stop - start)
            for p, c in param_list:
                sel = code == c
                if sel.any():
                    total += np.bincount(i[sel], weights=p.valence(d[sel]),
                                         minlength=stop - start)
            bvs[start:stop] = total
        if tree_c is not None:
            dist, _ = tree_c.query(points, k=1,
                                   distance_upper_bound=repulsion.r_excl_ang)
            excluded[start:stop] = dist < repulsion.r_excl_ang
    target = float(abs(probe_ox))
    mismatch = np.abs(bvs - target)
    spacing = tuple(float(v) for v in np.linalg.norm(box, axis=1) / shape)
    voxel = frame.volume_ang3 / n_points

    counts = {e: int((symbols == e).sum()) for e in anion_elements}
    notes = [
        f"probe {label}: V(x) is the sum of exp((R0 - d) / b) over every anion "
        f"image within {r_cut:g} Å of the grid point (r_cut_ang), one "
        f"parameter per anion element from {params.name}: " + "; ".join(
            f"{label}-{e}: R0 {p.r0:g} Å, b {p.b:g} Å, {p.source}, a contact "
            f"at r_cut carries {p.valence(r_cut):.3g} v.u."
            for e, p in parameters.items()),
        f"anions (oxidation state below 0): " + ", ".join(
            f"{e} {counts[e]}" for e in anion_elements)
        + f"; {int(with_param.sum())} of them carry a parameter",
        f"grid of {shape[0]} x {shape[1]} x {shape[2]} = {n_points} points, "
        f"spacing {spacing[0]:.4g}, {spacing[1]:.4g}, {spacing[2]:.4g} Å along "
        f"a, b, c ({h:g} Å asked for); point (i, j, k) at fractional "
        f"(i/{shape[0]}, j/{shape[1]}, k/{shape[2]}), the point of index 0 at "
        f"the box corner; voxel volume {voxel:.6g} Å^3",
        f"the mismatch is |V - {target:g}| in valence units: no energy scaling "
        "is applied, and no barrier can be read from it",
        f"lowest mismatch on the grid {float(mismatch.min()):.4g} v.u., "
        f"median {float(np.median(mismatch)):.4g} v.u.; V ranges "
        f"{float(bvs.min()):.4g} to {float(bvs.max()):.4g} v.u.",
    ]
    if missing:
        notes.append(
            f"no bond-valence parameter for {label} with "
            f"{', '.join(missing)} in {params.name}; those anions contribute "
            "nothing to V")
    estimated = [e for e, p in parameters.items() if not p.fitted]
    if estimated:
        notes.append("estimated, not fitted, parameters for " + ", ".join(
            f"{label}-{e} ({parameters[e].source})" for e in estimated))
    if repulsion is None:
        notes.append("no repulsion exclusion: every grid point is a candidate "
                     "position of the probe, those inside the model's atoms "
                     "included (their V is large and their mismatch with it)")
    else:
        notes += rep_notes
        notes.append(f"{int(excluded.sum())} of {n_points} grid points "
                     f"({excluded.mean():.4g} of the box) are excluded")
    return Landscape(
        probe=probe, probe_ox=probe_ox,
        bvs_vu=_frozen(bvs.reshape(tuple(shape))),
        mismatch_vu=_frozen(mismatch.reshape(tuple(shape))),
        excluded=None if excluded is None else _frozen(
            excluded.reshape(tuple(shape))),
        box_ang=frame.box_ang, origin_ang=frame.origin_ang,
        shape=tuple(int(v) for v in shape), grid_spacing_ang=h,
        spacing_ang=spacing, voxel_volume_ang3=voxel, r_cut_ang=r_cut,
        params_name=str(getattr(params, "name", "")),
        parameters={e: (float(p.r0), float(p.b), str(p.source))
                    for e, p in parameters.items()},
        anions_without_parameter=tuple(missing), repulsion=repulsion,
        repulsion_elements=rep_elements, elements=frame.elements,
        frac=frame.frac, frame_digest=bulk._frame_digest(frame),
        notes=tuple(notes))


def _check_landscape(landscape) -> Landscape:
    if not isinstance(landscape, Landscape):
        raise ValueError(f"a Landscape is needed, not "
                         f"{type(landscape).__name__}; bv_landscape builds one")
    return landscape


def accessible_fraction(landscape: Landscape, deltas_vu) -> np.ndarray:
    """The volume fraction of the box whose mismatch is at most each Delta.

    ``deltas_vu`` is a number or an array of thresholds in valence units;
    the result has its shape. Excluded points never count; the denominator
    is every grid point, so the fraction is of the box volume. Monotone
    non-decreasing in Delta by construction.
    """
    landscape = _check_landscape(landscape)
    deltas = np.asarray(deltas_vu, dtype=np.float64)
    if deltas.size and (not np.isfinite(deltas).all() or (deltas < 0).any()):
        raise ValueError("deltas_vu needs finite thresholds of 0 or more, in "
                         "valence units")
    values = landscape.sorted_mismatch()
    counts = np.searchsorted(values, deltas, side="right")
    return counts / landscape.n_points


def _label_periodic(mask: np.ndarray):
    """ndimage labels of ``mask`` (6-connectivity, not periodic), their
    number, and the edges (u, v, shift) that join a label on the last plane
    of an axis to the label on the first plane of the next cell."""
    from scipy import ndimage

    labels, n = ndimage.label(mask)
    shape = mask.shape
    us, vs, shifts = [], [], []
    for axis in range(3):
        first = np.take(labels, 0, axis=axis)
        last = np.take(labels, shape[axis] - 1, axis=axis)
        both = (first > 0) & (last > 0)
        if both.any():
            pair = np.unique(np.column_stack([last[both], first[both]]), axis=0)
            us.append(pair[:, 0].astype(np.int64) - 1)
            vs.append(pair[:, 1].astype(np.int64) - 1)
            s = np.zeros((pair.shape[0], 3), dtype=np.int64)
            s[:, axis] = 1
            shifts.append(s)
    if us:
        return (labels, int(n), np.concatenate(us), np.concatenate(vs),
                np.concatenate(shifts))
    empty = np.zeros(0, dtype=np.int64)
    return labels, int(n), empty, empty, np.zeros((0, 3), dtype=np.int64)


def _accessible_mask(landscape: Landscape, delta: float) -> np.ndarray:
    mask = landscape.mismatch_vu <= delta
    if landscape.excluded is not None:
        mask &= ~landscape.excluded
    return mask


def _spans_at(landscape: Landscape, delta: float) -> np.ndarray:
    """(3,) whether some accessible region at ``delta`` spans a, b, c."""
    labels, n, u, v, shift = _label_periodic(_accessible_mask(landscape, delta))
    if n == 0:
        return np.zeros(3, dtype=bool)
    _, _, bases = _periodic_components(n, u, v, shift)
    return _component_geometry(bases)[1].any(axis=0)


def _region_moments(flat: np.ndarray, region: np.ndarray, cell: np.ndarray,
                    shape: np.ndarray, box: np.ndarray, n_regions: int):
    """Per region, the count, the sum of the unwrapped positions and the
    sum of their outer products, over the grid points ``flat`` (flat
    indices) in regions ``region`` placed in cells ``cell`` (per point)."""
    count = np.bincount(region, minlength=n_regions).astype(np.float64)
    s1 = np.zeros((n_regions, 3))
    s1_wrapped = np.zeros((n_regions, 3))
    s2 = np.zeros((n_regions, 3, 3))
    for start in range(0, flat.size, _POINTS_PER_CHUNK):
        stop = min(flat.size, start + _POINTS_PER_CHUNK)
        i, j, k = np.unravel_index(flat[start:stop], tuple(shape))
        frac = np.column_stack([i / shape[0], j / shape[1], k / shape[2]])
        home = bulk._lattice(frac, box)
        cart = bulk._lattice(frac + cell[start:stop], box)
        reg = region[start:stop]
        for a in range(3):
            s1[:, a] += np.bincount(reg, weights=cart[:, a],
                                    minlength=n_regions)
            s1_wrapped[:, a] += np.bincount(reg, weights=home[:, a],
                                            minlength=n_regions)
            for b in range(a, 3):
                s2[:, a, b] += np.bincount(reg, weights=cart[:, a] * cart[:, b],
                                           minlength=n_regions)
    for a in range(3):
        for b in range(a + 1, 3):
            s2[:, b, a] = s2[:, a, b]
    return count, s1, s1_wrapped, s2


def _covariances(count, s1, s2, extra: np.ndarray) -> tuple:
    """(mean, covariance + extra) per region from the moment sums."""
    mean = s1 / count[:, None]
    cov = s2 / count[:, None, None] - mean[:, :, None] * mean[:, None, :]
    cov = 0.5 * (cov + np.transpose(cov, (0, 2, 1))) + extra
    return mean, cov


@dataclass(frozen=True, eq=False)
class AccessibleRegions:
    """The connected accessible regions of a landscape at one threshold.

    ``label_grid`` gives every grid point its region (-1 when not
    accessible), regions numbered largest first. Per region: ``n_points``,
    ``volume_ang3``, ``centroid_ang`` (the box origin plus the mean of the
    points, unwrapped into one copy for a finite region and as wrapped into
    the box for a spanning one), ``elongation``, ``mismatch_min_vu``,
    ``dimensionality``, ``spans``, ``copies``, and ``probe_atoms`` (rows of
    the probe element whose nearest grid point lies in the region).
    ``region_of_atom`` gives every atom the region of its nearest grid
    point. ``percolates`` says whether any region spans a, b, c.
    """

    delta_vu: float
    probe: str
    probe_ox: int
    label_grid: np.ndarray
    n_regions: int
    n_points: np.ndarray
    volume_ang3: np.ndarray
    centroid_ang: np.ndarray
    elongation: np.ndarray
    mismatch_min_vu: np.ndarray
    dimensionality: np.ndarray
    spans: np.ndarray
    copies: np.ndarray
    region_of_atom: np.ndarray
    probe_atoms: tuple
    accessible_fraction: float
    percolates: np.ndarray
    box_ang: np.ndarray
    origin_ang: np.ndarray
    shape: tuple[int, int, int]
    voxel_volume_ang3: float
    landscape_parameters: Mapping[str, object]
    notes: tuple[str, ...] = ()

    kind: str = "accessible region"

    @property
    def method_parameters(self) -> dict:
        return {**dict(self.landscape_parameters), "Delta (v.u.)": self.delta_vu}

    def as_rows(self) -> list[dict]:
        rows = []
        for r in range(self.n_regions):
            rows.append({
                "descriptor": self.kind, "region": r,
                "grid points": int(self.n_points[r]),
                "volume (Å^3)": float(self.volume_ang3[r]),
                "centroid x (Å)": float(self.centroid_ang[r, 0]),
                "centroid y (Å)": float(self.centroid_ang[r, 1]),
                "centroid z (Å)": float(self.centroid_ang[r, 2]),
                "elongation": float(self.elongation[r]),
                "lowest mismatch (v.u.)": float(self.mismatch_min_vu[r]),
                "dimensionality": int(self.dimensionality[r]),
                "spans a": bool(self.spans[r, 0]),
                "spans b": bool(self.spans[r, 1]),
                "spans c": bool(self.spans[r, 2]),
                "interpenetrating copies": int(self.copies[r]),
                f"{self.probe} atoms inside": int(self.probe_atoms[r].size)})
        return rows


def accessible_regions(landscape: Landscape, delta_vu: float
                       ) -> AccessibleRegions:
    """The connected regions where the mismatch is at most ``delta_vu``
    (6-connectivity, periodic), with their volume, centroid, elongation,
    dimensionality, spans and the probe atoms inside (module docstring)."""
    landscape = _check_landscape(landscape)
    delta = _nonnegative(delta_vu, "delta_vu")
    shape = np.array(landscape.shape, dtype=np.int64)
    box = landscape.box_ang
    n_points = landscape.n_points
    mask = _accessible_mask(landscape, delta)
    labels, n_lab, u, v, shift = _label_periodic(mask)
    probe_rows = np.flatnonzero(landscape.elements == landscape.probe)
    nearest = np.mod(np.rint(landscape.frac * shape).astype(np.int64), shape)
    nearest_flat = np.ravel_multi_index(
        (nearest[:, 0], nearest[:, 1], nearest[:, 2]), tuple(shape))
    base_notes = [f"Delta = {delta:g} v.u.: accessible where |V - "
                  f"{landscape.target_vu:g}| <= Delta"
                  + ("" if landscape.excluded is None else
                     " and not excluded by the repulsion"),
                  _CONNECTIVITY_TEXT]
    if n_lab == 0:
        empty = np.zeros(0)
        return AccessibleRegions(
            delta_vu=delta, probe=landscape.probe, probe_ox=landscape.probe_ox,
            label_grid=_frozen(np.full(tuple(shape), -1, dtype=np.int32)),
            n_regions=0, n_points=_frozen(np.zeros(0, dtype=np.int64)),
            volume_ang3=_frozen(empty), centroid_ang=_frozen(np.zeros((0, 3))),
            elongation=_frozen(empty), mismatch_min_vu=_frozen(empty),
            dimensionality=_frozen(np.zeros(0, dtype=np.int64)),
            spans=_frozen(np.zeros((0, 3), dtype=bool)),
            copies=_frozen(np.zeros(0, dtype=np.int64)),
            region_of_atom=_frozen(np.full(landscape.elements.shape[0], -1,
                                           dtype=np.int64)),
            probe_atoms=(), accessible_fraction=0.0,
            percolates=_frozen(np.zeros(3, dtype=bool)), box_ang=box,
            origin_ang=landscape.origin_ang, shape=landscape.shape,
            voxel_volume_ang3=landscape.voxel_volume_ang3,
            landscape_parameters=landscape.method_parameters,
            notes=tuple(base_notes + ["no grid point is accessible at this "
                                      "Delta"]))
    comp, cell_of_label, bases = _periodic_components(n_lab, u, v, shift)
    n_regions = len(bases)
    dimensionality, spans, copies = _component_geometry(bases)

    flat_labels = labels.ravel()
    inside = np.flatnonzero(flat_labels > 0)
    lab_in = flat_labels[inside] - 1
    reg_in = comp[lab_in]
    order = np.argsort(reg_in, kind="stable")
    reg_sorted = reg_in[order]
    starts = np.searchsorted(reg_sorted, np.arange(n_regions))
    mis_sorted = landscape.mismatch_vu.ravel()[inside][order]
    mismatch_min = np.minimum.reduceat(mis_sorted, starts)
    first_flat = inside[order][starts]
    count, s1, s1_wrapped, s2 = _region_moments(
        inside, reg_in, cell_of_label[lab_in], shape, box, n_regions)
    mean, cov = _covariances(count, s1, s2, np.zeros((3, 3)))
    voxel_cov = _voxel_covariance(box, shape)
    tensors = [_spanning_tensor(c, b, box, voxel_cov)[0]
               for c, b in zip(cov, bases)]
    elongation = np.array([_elongation(t) for t in tensors])
    centroid = np.where(dimensionality[:, None] == 0, mean,
                        s1_wrapped / count[:, None])

    rank = np.lexsort((first_flat, mismatch_min, -count))
    relabel = np.empty(n_regions, dtype=np.int64)
    relabel[rank] = np.arange(n_regions)
    region_flat = np.full(n_points, -1, dtype=np.int32)
    region_flat[inside] = relabel[reg_in]
    region_of_atom = region_flat[nearest_flat].astype(np.int64)
    probe_atoms = tuple(
        _frozen(probe_rows[region_of_atom[probe_rows] == r])
        for r in range(n_regions))
    volume = count[rank] * landscape.voxel_volume_ang3
    percolates = spans[rank].any(axis=0)
    notes = base_notes + [
        f"{n_regions} region(s) over {inside.size} accessible grid points "
        f"({inside.size / n_points:.6g} of the box); the largest holds "
        f"{int(count[rank][0])} points, {volume[0]:.6g} Å^3; regions spanning "
        f"an axis: {int(spans.any(axis=1).sum())} (axes spanned by any: "
        f"{_spans_text(percolates)})",
        f"{int(sum(a.size for a in probe_atoms))} of {probe_rows.size} "
        f"{landscape.probe} atoms have their nearest grid point in an "
        "accessible region",
        "elongation: sqrt of the largest over the smallest eigenvalue of the "
        "gyration tensor of the region's points plus one voxel's own tensor; "
        "a finite region is unwrapped into its one connected copy; a region "
        "spanning one axis takes the spread perpendicular to its translation "
        "T and |T|^2 / 12 along it (one period), and its centroid is that of "
        "its points as wrapped into the box; a region spanning two or three "
        "axes has no elongation (NaN)"]
    split = np.flatnonzero(copies[rank] > 1)
    if split.size:
        notes.append(
            f"{split.size} region(s) stand for several interpenetrating "
            "pieces that a box translation maps onto each other (copies > 1): "
            + ", ".join(f"region {r} x{int(copies[rank][r])}" for r in split[:10]))
    return AccessibleRegions(
        delta_vu=delta, probe=landscape.probe, probe_ox=landscape.probe_ox,
        label_grid=_frozen(region_flat.reshape(tuple(shape))),
        n_regions=n_regions, n_points=_frozen(count[rank].astype(np.int64)),
        volume_ang3=_frozen(volume),
        centroid_ang=_frozen(landscape.origin_ang + centroid[rank]),
        elongation=_frozen(elongation[rank]),
        mismatch_min_vu=_frozen(mismatch_min[rank]),
        dimensionality=_frozen(dimensionality[rank]), spans=_frozen(spans[rank]),
        copies=_frozen(copies[rank]), region_of_atom=_frozen(region_of_atom),
        probe_atoms=probe_atoms, accessible_fraction=inside.size / n_points,
        percolates=_frozen(percolates), box_ang=box,
        origin_ang=landscape.origin_ang, shape=landscape.shape,
        voxel_volume_ang3=landscape.voxel_volume_ang3,
        landscape_parameters=landscape.method_parameters, notes=tuple(notes))


@dataclass(frozen=True, eq=False)
class PercolationThresholds:
    """The smallest Delta at which an accessible region spans each axis.

    ``delta_vu`` (3,) per box axis, NaN when no path spans the axis even
    with every non-excluded point accessible; ``fraction_at`` the accessible
    fraction at each threshold; ``delta_any_vu`` the smallest of the three.
    """

    probe: str
    probe_ox: int
    delta_vu: np.ndarray
    fraction_at: np.ndarray
    n_evaluations: int
    landscape_parameters: Mapping[str, object]
    notes: tuple[str, ...] = ()

    @property
    def delta_any_vu(self) -> float:
        finite = self.delta_vu[np.isfinite(self.delta_vu)]
        return float(finite.min()) if finite.size else math.nan

    @property
    def method_parameters(self) -> dict:
        return dict(self.landscape_parameters)

    def as_rows(self) -> list[dict]:
        return [{"descriptor": "percolation threshold", "axis": AXES[k],
                 "Delta (v.u.)": float(self.delta_vu[k]),
                 "accessible fraction at Delta": float(self.fraction_at[k])}
                for k in range(3)]


def percolation_thresholds(landscape: Landscape) -> PercolationThresholds:
    """The smallest mismatch threshold at which some accessible region spans
    a, b and c, by bisection over the sorted mismatch values of the grid
    (module docstring). The result is a grid value, exact for this grid."""
    landscape = _check_landscape(landscape)
    values = landscape.sorted_mismatch()
    if values.size == 0:
        raise ValueError("every grid point is excluded by the repulsion; no "
                         "threshold exists")
    top = _spans_at(landscape, float(values[-1]))
    evaluations = 1
    lo = np.full(3, -1, dtype=np.int64)
    hi = np.full(3, values.size - 1, dtype=np.int64)
    while True:
        width = np.where(top, hi - lo, 0)
        k = int(np.argmax(width))
        if width[k] <= 1:
            break
        mid = int((lo[k] + hi[k]) // 2)
        spans = _spans_at(landscape, float(values[mid]))
        evaluations += 1
        for axis in range(3):
            if not top[axis]:
                continue
            if spans[axis]:
                hi[axis] = min(hi[axis], mid)
            else:
                lo[axis] = max(lo[axis], mid)
    delta = np.where(top, values[np.minimum(hi, values.size - 1)], np.nan)
    fraction = np.full(3, np.nan)
    for axis in range(3):
        if top[axis]:
            fraction[axis] = np.searchsorted(values, delta[axis],
                                             side="right") / landscape.n_points
    notes = [
        "the smallest Delta at which some region of |V - "
        f"{landscape.target_vu:g}| <= Delta spans the axis, found by bisection "
        f"over the {values.size} sorted mismatch values of the non-excluded "
        f"grid points in {evaluations} labellings; each threshold is a value "
        "of the grid",
        _CONNECTIVITY_TEXT,
        "thresholds: " + ", ".join(
            f"{AXES[k]} {delta[k]:.6g} v.u. (accessible fraction "
            f"{fraction[k]:.6g})" if top[k] else f"{AXES[k]} none"
            for k in range(3))]
    blocked = [AXES[k] for k in range(3) if not top[k]]
    if blocked:
        notes.append(
            f"no accessible region spans {', '.join(blocked)} even with every "
            "non-excluded grid point accessible"
            + (": the repulsion exclusion closes every path along it"
               if landscape.excluded is not None else
               ", which happens only when the grid holds no point along "
               "that axis"))
    return PercolationThresholds(
        probe=landscape.probe, probe_ox=landscape.probe_ox,
        delta_vu=_frozen(delta), fraction_at=_frozen(fraction),
        n_evaluations=evaluations,
        landscape_parameters=landscape.method_parameters, notes=tuple(notes))


# ---------------------------------------------------------------------------
# exports to volume.Grid, for the isosurface and section code
# ---------------------------------------------------------------------------

def _grid_of(values: np.ndarray, box: np.ndarray, name: str, units: str,
             notes: Sequence[str]):
    from . import readers, volume

    return volume.Grid(np.array(values, dtype=np.float64),
                       readers.cell_from_vectors(box), name=name, units=units,
                       source=None, notes=list(notes))


def landscape_grid(landscape: Landscape, field: str = "mismatch"):
    """The landscape as a :class:`facet.core.volume.Grid`: ``'mismatch'``
    (|V - n|, v.u.), ``'bvs'`` (V, v.u.) or ``'excluded'`` (1 where
    excluded, 0 elsewhere). The isosurface at a level Delta of the mismatch
    grid bounds the accessible volume at Delta."""
    landscape = _check_landscape(landscape)
    label = landscape.probe_label
    if field == "mismatch":
        return _grid_of(landscape.mismatch_vu, landscape.box_ang,
                        f"bond-valence mismatch of {label}", "v.u.",
                        landscape.notes)
    if field == "bvs":
        return _grid_of(landscape.bvs_vu, landscape.box_ang,
                        f"bond-valence sum of {label}", "v.u.", landscape.notes)
    if field == "excluded":
        if landscape.excluded is None:
            raise ValueError("this landscape has no repulsion exclusion")
        return _grid_of(landscape.excluded, landscape.box_ang,
                        f"repulsion exclusion for {label}", "1",
                        landscape.notes)
    raise ValueError(f"field {field!r}: one of 'mismatch', 'bvs', 'excluded'")


def regions_grid(regions):
    """The region labels as a :class:`facet.core.volume.Grid`: region r is
    the value r + 1, and 0 lies outside every region, so the isosurface at
    0.5 bounds every region at once."""
    if not hasattr(regions, "label_grid"):
        raise ValueError("accessible_regions or void_regions results are "
                         "needed")
    return _grid_of(regions.label_grid + 1, regions.box_ang,
                    f"{regions.kind}s (region id + 1)", "1", regions.notes)


def region_indicator_grid(regions, region: int):
    """1 inside region ``region`` and 0 elsewhere, as a
    :class:`facet.core.volume.Grid`; the isosurface at 0.5 is its boundary."""
    if not hasattr(regions, "label_grid"):
        raise ValueError("accessible_regions or void_regions results are "
                         "needed")
    r = _whole(region, "region", 0)
    if r >= regions.n_regions:
        raise ValueError(f"region {r} is outside 0 .. {regions.n_regions - 1}")
    return _grid_of(regions.label_grid == r, regions.box_ang,
                    f"{regions.kind} {r}", "1", regions.notes)


# ---------------------------------------------------------------------------
# B. structurally, by modifier density
# ---------------------------------------------------------------------------

def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


@dataclass(frozen=True)
class MeasuredCutoffs:
    """Cutoffs read from the first minimum of the partial g(r) of one frame.

    ``cutoffs_ang`` and ``cutoff_sources`` are what :func:`modifier_density`
    and :func:`facet.core.md_network.graph_from_pairs` take; ``minima`` holds
    each pair's :class:`facet.core.glass.RdfMinimum`; ``missing`` names the
    pairs whose g(r) has no such point, with the reason the rule gives.
    """

    cutoffs_ang: Mapping[tuple[str, str], float]
    cutoff_sources: Mapping[tuple[str, str], str]
    minima: Mapping[tuple[str, str], glass.RdfMinimum]
    missing: Mapping[tuple[str, str], str]
    method: str
    r_max_ang: float
    dr_ang: float


def measure_cutoffs(frame: Frame, pairs, element_pairs, method: glass.MinimumMethod,
                    *, r_max_ang: float, dr_ang: float = glass.RDF_DR_ANG
                    ) -> MeasuredCutoffs:
    """The first minimum of g_ab(r) of this frame for each (a, b) in
    ``element_pairs``, as a cutoff with its source.

    ``pairs`` is a :class:`facet.core.bulk.PairTable` (or the blocks of one
    search, each once) searched to at least ``r_max_ang + dr_ang``; give a
    table rather than a generator when the same pairs feed
    :func:`modifier_density` afterwards. ``method`` is the rule
    (:class:`facet.core.glass.MinimumMethod`, no default), applied to the
    g(r) of this one frame with its own counting errors; a minimum read
    from the frame-averaged g(r) of a trajectory is the better-sampled one,
    and ``analyse_trajectory`` gives those (``GlassResult.cutoffs_ang``).
    """
    frame = _check_frame(frame)
    if not isinstance(method, glass.MinimumMethod):
        raise ValueError("method needs a glass.MinimumMethod (no default rule)")
    if isinstance(element_pairs, (str, bytes, Mapping)):
        raise ValueError("element_pairs needs a collection of (element, "
                         "element) pairs")
    wanted = []
    for item in element_pairs:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(f"element pair {item!r}: a (element, element) "
                             "pair is needed")
        key = _pair_key(validate_symbol(str(item[0])),
                        validate_symbol(str(item[1])))
        if key not in wanted:
            wanted.append(key)
    if not wanted:
        raise ValueError("element_pairs is empty")
    partials = glass.partial_rdf(frame, pairs, r_max_ang=r_max_ang,
                                 dr_ang=dr_ang)
    cutoffs: dict[tuple[str, str], float] = {}
    sources: dict[tuple[str, str], str] = {}
    minima: dict[tuple[str, str], glass.RdfMinimum] = {}
    missing: dict[tuple[str, str], str] = {}
    for key in wanted:
        if key not in partials:
            raise ValueError(f"the frame holds no {key[0]}-{key[1]} pair: "
                             f"its elements are {', '.join(frame.species)}")
        p = partials[key]
        found = glass.first_minimum(p.r_ang, p.g, method, pair=key,
                                    level=p.uncorrelated_level,
                                    pairs_per_unit_g=p.pair_counts_per_unit_g)
        minima[key] = found
        if found.found:
            cutoffs[key] = float(found.r_ang)
            floor = found.floor_r_ang
            sources[key] = (
                f"first minimum of the partial g_{key[0]}{key[1]}(r) of this "
                f"frame, {found.r_ang:.6g} Å (floor {floor[0]:.6g}-"
                f"{floor[1]:.6g} Å; {found.method})")
        else:
            missing[key] = found.reason
    return MeasuredCutoffs(cutoffs_ang=cutoffs, cutoff_sources=sources,
                           minima=minima, missing=missing,
                           method=method.describe(),
                           r_max_ang=_positive(r_max_ang, "r_max_ang"),
                           dr_ang=_positive(dr_ang, "dr_ang"))


@dataclass(frozen=True, eq=False)
class ModifierDensity:
    """Modifier counts around the anions of one frame, the modifier-rich
    anions and their clusters.

    ``count`` (N,) is the number of modifier atoms within the cutoff of each
    anion row (-1 on other rows); ``anion_count`` the anions within the
    cutoff of each modifier row (-1 elsewhere); ``rich`` marks the anions
    with ``count >= k_rich``; ``label`` is the cluster of each atom (-1
    outside the clusters), ``clusters`` the
    :class:`facet.core.md_network.Components` of the cluster graph (None when
    no anion is rich). Per cluster, largest first: ``cluster_sizes`` (rich
    anions and their modifiers), ``cluster_rich`` (rich anions),
    ``cluster_dimensionality`` and ``cluster_spans``. ``speciation`` maps
    each category of :data:`facet.core.glass.SPECIATION_KEYS` to (rich
    anions, anions) in it, or is None when no bonds were given.
    """

    modifiers: frozenset
    anions: frozenset
    k_rich: int
    count: np.ndarray
    anion_count: np.ndarray
    rich: np.ndarray
    label: np.ndarray
    clusters: Components | None
    n_anions: int
    n_rich: int
    n_modifiers: int
    cluster_sizes: np.ndarray
    cluster_rich: np.ndarray
    cluster_dimensionality: np.ndarray
    cluster_spans: np.ndarray
    modifiers_in_clusters: int
    speciation: Mapping[str, tuple[int, int]] | None
    cutoffs_ang: Mapping
    cutoff_sources: Mapping
    formers: frozenset | None
    v_bond_vu: float | None
    notes: tuple[str, ...] = ()

    @property
    def fraction_rich(self) -> float:
        return self.n_rich / self.n_anions

    @property
    def n_clusters(self) -> int:
        return int(self.cluster_sizes.shape[0])

    @property
    def largest_cluster_fraction(self) -> float:
        """The largest piece's share of the cluster nodes (rich anions and
        their modifiers); 0 when there is no cluster."""
        return self.clusters.largest_fraction if self.clusters is not None \
            else 0.0

    @property
    def largest_cluster_rich_fraction(self) -> float:
        """The rich anions in the largest cluster over all rich anions; 0
        when none is rich."""
        if self.n_rich == 0:
            return 0.0
        return float(self.cluster_rich[0]) / self.n_rich

    @property
    def percolates(self) -> np.ndarray:
        """(3,) whether any cluster spans a, b, c."""
        if self.cluster_spans.shape[0] == 0:
            return np.zeros(3, dtype=bool)
        return self.cluster_spans.any(axis=0)

    def count_distribution(self) -> dict[int, int]:
        """count -> number of anions with that many modifiers."""
        values, counts = np.unique(self.count[self.count >= 0],
                                   return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}

    @property
    def method_parameters(self) -> dict:
        out = {"modifiers": tuple(sorted(self.modifiers)),
               "anions": tuple(sorted(self.anions)), "k_rich": self.k_rich}
        for (a, b), value in sorted(self.cutoffs_ang.items()):
            out[f"cutoff {a}-{b} (Å)"] = value
        if self.formers is not None:
            out["formers"] = tuple(sorted(self.formers))
        if self.v_bond_vu is not None:
            out["v_bond (v.u.)"] = self.v_bond_vu
        return out

    def as_rows(self) -> list[dict]:
        rows = []
        for c in range(self.n_clusters):
            rows.append({"descriptor": "modifier cluster", "cluster": c,
                         "nodes": int(self.cluster_sizes[c]),
                         "rich anions": int(self.cluster_rich[c]),
                         "modifiers": int(self.cluster_sizes[c]
                                          - self.cluster_rich[c]),
                         "dimensionality": int(self.cluster_dimensionality[c]),
                         "spans a": bool(self.cluster_spans[c, 0]),
                         "spans b": bool(self.cluster_spans[c, 1]),
                         "spans c": bool(self.cluster_spans[c, 2])})
        return rows


def modifier_density(frame: Frame, pairs, *, modifiers, anions, cutoffs_ang: Mapping,
                     cutoff_sources: Mapping, k_rich: int, bonds=None,
                     formers=None) -> ModifierDensity:
    """Modifier atoms within the cutoff of every anion, the anions with
    ``k_rich`` or more, and the clusters they form (module docstring).

    ``pairs`` is a :class:`facet.core.bulk.PairTable` or the blocks of one
    ``bulk.iter_pairs`` search of this frame, searched at least to the
    largest cutoff. ``cutoffs_ang`` maps every (modifier, anion) pair present
    in the frame to its cutoff in Å, each with a source in
    ``cutoff_sources`` (:func:`measure_cutoffs` gives both); a pair that is
    not modifier-anion is refused, and so is a present pair without a
    cutoff. With ``bonds`` (:class:`facet.core.bulk.Bonds`) and ``formers``,
    the speciation of the anions is crossed with their richness.
    """
    frame = _check_frame(frame)
    mods = _symbols(modifiers, "modifiers")
    ans = _symbols(anions, "anions")
    shared = sorted(mods & ans)
    if shared:
        raise ValueError(f"{', '.join(shared)} is named both as a modifier and "
                         "as an anion")
    k = _whole(k_rich, "k_rich", 1)
    symbols = frame.elements
    n = frame.n_atoms
    present = set(frame.species)
    if not isinstance(cutoffs_ang, Mapping):
        raise ValueError("cutoffs_ang needs a mapping (modifier, anion) -> "
                         "cutoff in Å; measure_cutoffs gives one")
    given = set()
    for key in cutoffs_ang:
        if not (isinstance(key, tuple) and len(key) == 2):
            raise ValueError(f"cutoff key {key!r}: a (modifier, anion) pair "
                             "is needed")
        a, b = validate_symbol(str(key[0])), validate_symbol(str(key[1]))
        if a in mods and b in ans:
            given.add((a, b))
        elif b in mods and a in ans:
            given.add((b, a))
        else:
            raise ValueError(f"the {a}-{b} cutoff does not join a modifier "
                             f"({', '.join(sorted(mods))}) to an anion "
                             f"({', '.join(sorted(ans))})")
    needed = [(m, a) for m in sorted(mods & present) for a in sorted(ans & present)]
    if not needed:
        raise ValueError(
            f"the frame holds no modifier-anion pair of the elements named "
            f"(modifiers {', '.join(sorted(mods))}, anions "
            f"{', '.join(sorted(ans))}; the frame holds "
            f"{', '.join(frame.species)})")
    missing = [p for p in needed if p not in given]
    if missing:
        raise ValueError(
            "no cutoff for " + ", ".join(f"{m}-{a}" for m, a in missing)
            + "; every modifier-anion pair of the frame needs one "
            "(measure_cutoffs reads them from the partial g(r) of the frame)")
    graph = graph_from_pairs(frame, pairs, cutoffs_ang, cutoff_sources,
                             elements=mods | ans)
    is_anion_row = np.isin(symbols, sorted(ans))
    is_mod_row = np.isin(symbols, sorted(mods))
    degree = graph.degree
    count = np.where(is_anion_row, degree, -1).astype(np.int64)
    anion_count = np.where(is_mod_row, degree, -1).astype(np.int64)
    rich = is_anion_row & (degree >= k)
    n_anions = int(is_anion_row.sum())
    n_mods = int(is_mod_row.sum())
    n_rich = int(rich.sum())

    label = np.full(n, -1, dtype=np.int64)
    comps = None
    sizes = np.zeros(0, dtype=np.int64)
    cluster_rich = np.zeros(0, dtype=np.int64)
    dimensionality = np.zeros(0, dtype=np.int64)
    spans = np.zeros((0, 3), dtype=bool)
    modifiers_in = 0
    if n_rich:
        u, v, shift = graph.u, graph.v, graph.shift
        keep = rich[u] | rich[v]
        nodes = rich.copy()
        nodes[u[keep]] = True
        nodes[v[keep]] = True
        definition = (
            f"modifier clusters: anions ({', '.join(sorted(ans))}) with at "
            f"least {k} modifier atoms ({', '.join(sorted(mods))}) within the "
            "cutoff, and the modifiers within the cutoff of one of them, one "
            "edge per such contact; " + graph.definition)
        sub = _make_graph(frame, nodes, u[keep], v[keep], shift[keep],
                          np.ones(int(keep.sum()), dtype=np.int64),
                          source="distance", definition=definition,
                          node_elements=mods | ans, joinable=graph.joinable,
                          cutoffs=graph.cutoffs_ang,
                          cutoff_sources=graph.cutoff_sources, notes=graph.notes)
        comps = components(sub)
        label = comps.label
        sizes = comps.sizes
        dimensionality = comps.dimensionality
        spans = comps.spans
        cluster_rich = np.bincount(label[rich], minlength=comps.n_components
                                   ).astype(np.int64)
        modifiers_in = int((nodes & is_mod_row).sum())

    speciation = None
    former_set = None
    v_bond = None
    spec_notes: list[str] = []
    if bonds is not None:
        if not isinstance(bonds, bulk.Bonds):
            raise ValueError(f"bonds needs a bulk.Bonds (bulk.bonds_at gives "
                             f"one), not {type(bonds).__name__}")
        if formers is None:
            raise ValueError("formers has no default: the speciation of an "
                             "anion counts its bonded network formers, so the "
                             "former set is needed with the bonds")
        former_set = _symbols(formers, "formers")
        cation = bonds.cation.astype(np.int64)
        anion = bonds.anion.astype(np.int64)
        if cation.size and (int(cation.max()) >= n or int(anion.max()) >= n):
            raise ValueError("the bonds name atom rows beyond this frame; they "
                             "belong to another frame")
        to_former = np.isin(symbols[cation], sorted(former_set))
        n_formers = np.bincount(anion[to_former], minlength=n)
        category = np.minimum(n_formers, 3)
        speciation = {}
        for index, key in enumerate(glass.SPECIATION_KEYS):
            in_cat = is_anion_row & (category == index)
            speciation[key] = (int((in_cat & rich).sum()), int(in_cat.sum()))
        v_bond = float(bonds.v_bond_vu)
        parts = []
        for key, (r, t) in speciation.items():
            if t:
                parts.append(f"{key} {r} of {t} ({r / t:.4g})")
        spec_notes.append(
            f"speciation by bonded formers ({', '.join(sorted(former_set))}; "
            f"bonds at v_bond {v_bond:g} v.u.): modifier-rich anions per "
            "category, " + "; ".join(parts))
        if n_rich:
            spec_notes.append(
                "of the modifier-rich anions, " + ", ".join(
                    f"{speciation[key][0] / n_rich:.4g} are {key}"
                    for key in glass.SPECIATION_KEYS if speciation[key][0]))
    elif formers is not None:
        spec_notes.append("formers were given without bonds, so no "
                          "speciation is crossed with the modifier counts")

    mean_count = float(count[is_anion_row].mean()) if n_anions else math.nan
    values, numbers = np.unique(count[is_anion_row], return_counts=True)
    notes = [
        f"modifier count: {', '.join(sorted(mods))} atoms within the cutoff "
        f"of each {', '.join(sorted(ans))} atom (d <= cutoff; " + "; ".join(
            f"{a}-{b} {value:g} Å, {graph.cutoff_sources[(a, b)]}"
            for (a, b), value in sorted(graph.cutoffs_ang.items())) + ")",
        f"{n_anions} anions, mean count {mean_count:.4g}; {n_rich} "
        f"({n_rich / n_anions:.4g}) hold {k} or more (k_rich) and are "
        "modifier-rich; anions per count: " + ", ".join(
            f"{int(v)}: {int(c)}" for v, c in zip(values, numbers)),
    ]
    if comps is not None:
        notes.append(
            f"{comps.n_components} cluster(s) of modifier-rich anions and "
            f"their modifiers ({modifiers_in} of {n_mods} modifier atoms are "
            f"in one); the largest holds {int(cluster_rich[0])} rich anions "
            f"({cluster_rich[0] / n_rich:.4g} of them) and "
            f"{comps.largest_fraction:.4g} of the cluster nodes; axes spanned "
            f"by any cluster: {_spans_text(spans.any(axis=0))}")
        notes += [f"clusters: {note}" for note in comps.notes
                  if not note.startswith("graph:")]
    else:
        notes.append(f"no anion holds {k} or more modifiers, so there is no "
                     "cluster")
    notes += spec_notes
    notes += [f"contacts: {note}" for note in graph.notes]
    return ModifierDensity(
        modifiers=mods, anions=ans, k_rich=k, count=_frozen(count),
        anion_count=_frozen(anion_count), rich=_frozen(rich),
        label=_frozen(label), clusters=comps, n_anions=n_anions, n_rich=n_rich,
        n_modifiers=n_mods, cluster_sizes=_frozen(sizes),
        cluster_rich=_frozen(cluster_rich),
        cluster_dimensionality=_frozen(dimensionality), cluster_spans=_frozen(spans),
        modifiers_in_clusters=modifiers_in, speciation=speciation,
        cutoffs_ang=dict(graph.cutoffs_ang),
        cutoff_sources=dict(graph.cutoff_sources), formers=former_set,
        v_bond_vu=v_bond, notes=tuple(notes))


# ---------------------------------------------------------------------------
# C. structurally, by voids
# ---------------------------------------------------------------------------

def _mark_spheres(centre_frac: np.ndarray, radii: np.ndarray, region: np.ndarray,
                  box: np.ndarray, shape: np.ndarray, label_flat: np.ndarray,
                  depth_flat: np.ndarray) -> None:
    """Give every grid point strictly inside a sphere the region of the
    sphere it lies deepest inside (the largest r - d), in place.

    The spheres are taken largest first, in chunks of about
    :data:`_SPHERE_WORK_PER_CHUNK` (sphere, grid offset) tests; the offsets
    tried are those of ``md_order._stencil`` for the chunk's largest radius,
    and every (sphere, point) pair kept is tested exactly.
    """
    widths = _widths(box)
    order = np.argsort(-radii, kind="stable")
    start = 0
    while start < order.size:
        r_max = float(radii[order[start]])
        offsets = md_order._stencil(box, shape, widths, r_max)
        per_chunk = max(1, _SPHERE_WORK_PER_CHUNK // offsets.shape[0])
        chunk = order[start:start + per_chunk]
        start += per_chunk
        u = centre_frac[chunk] * shape
        base = np.floor(u).astype(np.int64)
        t_xyz = ((u - base) / shape) @ box
        offset_xyz = (offsets / shape) @ box
        vec = offset_xyz[None, :, :] - t_xyz[:, None, :]
        d2 = np.einsum("sok,sok->so", vec, vec)
        inside = d2 < (radii[chunk] ** 2)[:, None]
        s_idx, o_idx = np.nonzero(inside)
        if not s_idx.size:
            continue
        depth = radii[chunk][s_idx] - np.sqrt(d2[s_idx, o_idx])
        index = np.mod(base[s_idx] + offsets[o_idx], shape)
        flat = np.ravel_multi_index((index[:, 0], index[:, 1], index[:, 2]),
                                    tuple(shape))
        np.maximum.at(depth_flat, flat, depth)
        win = depth_flat[flat] == depth
        label_flat[flat[win]] = region[chunk][s_idx[win]]


@dataclass(frozen=True, eq=False)
class VoidRegions:
    """Void regions of one frame: overlapping empty spheres grouped.

    ``sphere_rows`` indexes the :class:`facet.core.md_order.EmptySpheres`
    arrays, one row per distinct void kept (radius above 0 and at least the
    probe radius; coincident centres merged); ``sphere_region`` and
    ``sphere_cell`` give each kept sphere its region and the cell that places
    it in one connected copy. Per region, largest union volume first:
    ``n_spheres``, ``volume_sum_ang3`` (spheres summed, overlaps counted),
    ``volume_union_ang3`` (on the grid), ``centroid_ang`` (box origin plus the
    volume-weighted mean of the centres, unwrapped for a finite region and
    as wrapped into the box for a spanning one), ``elongation``,
    ``extent_ang`` (farthest sphere surface to farthest along the principal
    axis; |T| for a region spanning one axis; NaN beyond), ``radius_max_ang``,
    ``dimensionality``, ``spans``, ``copies`` and
    ``lining_atoms`` (rows). ``label_grid`` gives every grid point its
    region (-1 outside every sphere); ``region_of_atom`` gives every atom
    the region whose sphere it is nearest to among those it lines (-1 when
    it lines none). ``volume_ang3`` is the union volume, the one
    :func:`select_regions` compares.
    """

    probe_radius_ang: float
    grid_spacing_ang: float
    spacing_ang: tuple[float, float, float]
    lining_distance_ang: float
    sphere_rows: np.ndarray
    sphere_region: np.ndarray
    sphere_cell: np.ndarray
    n_regions: int
    n_spheres: np.ndarray
    volume_sum_ang3: np.ndarray
    volume_union_ang3: np.ndarray
    centroid_ang: np.ndarray
    elongation: np.ndarray
    extent_ang: np.ndarray
    radius_max_ang: np.ndarray
    dimensionality: np.ndarray
    spans: np.ndarray
    copies: np.ndarray
    label_grid: np.ndarray
    region_of_atom: np.ndarray
    lining_atoms: tuple
    void_fraction: float
    percolates: np.ndarray
    box_ang: np.ndarray
    origin_ang: np.ndarray
    shape: tuple[int, int, int]
    voxel_volume_ang3: float
    radii_ang: Mapping[str, float]
    radii_source: str
    n_left_out: int
    n_coincident: int
    notes: tuple[str, ...] = ()

    kind: str = "void region"

    @property
    def volume_ang3(self) -> np.ndarray:
        return self.volume_union_ang3

    @property
    def method_parameters(self) -> dict:
        return {"probe radius (Å)": self.probe_radius_ang,
                "grid spacing requested (Å)": self.grid_spacing_ang,
                "grid spacing (Å)": self.spacing_ang,
                "grid shape": self.shape,
                "lining distance (Å)": self.lining_distance_ang,
                "radii (Å)": dict(self.radii_ang),
                "radii source": self.radii_source,
                "coincidence tolerance (mean spacings)":
                    md_order.DEGENERATE_REL_TOL}

    def as_rows(self) -> list[dict]:
        rows = []
        for r in range(self.n_regions):
            rows.append({
                "descriptor": self.kind, "region": r,
                "spheres": int(self.n_spheres[r]),
                "union volume (Å^3)": float(self.volume_union_ang3[r]),
                "summed sphere volume (Å^3)": float(self.volume_sum_ang3[r]),
                "centroid x (Å)": float(self.centroid_ang[r, 0]),
                "centroid y (Å)": float(self.centroid_ang[r, 1]),
                "centroid z (Å)": float(self.centroid_ang[r, 2]),
                "elongation": float(self.elongation[r]),
                "extent (Å)": float(self.extent_ang[r]),
                "largest sphere radius (Å)": float(self.radius_max_ang[r]),
                "dimensionality": int(self.dimensionality[r]),
                "spans a": bool(self.spans[r, 0]),
                "spans b": bool(self.spans[r, 1]),
                "spans c": bool(self.spans[r, 2]),
                "interpenetrating copies": int(self.copies[r]),
                "lining atoms": int(self.lining_atoms[r].size)})
        return rows


def _reduce_by_group(group: np.ndarray, values: np.ndarray, n_groups: int,
                     reducer) -> np.ndarray:
    """``reducer.reduceat`` of ``values`` within each group (every group
    holds at least one member)."""
    order = np.argsort(group, kind="stable")
    starts = np.searchsorted(group[order], np.arange(n_groups))
    return reducer.reduceat(values[order], starts)


def void_regions(frame: Frame, spheres: EmptySpheres, *, probe_radius_ang: float,
                 grid_spacing_ang: float, lining_distance_ang: float
                 ) -> VoidRegions:
    """Group the empty spheres of ``spheres`` (``md_order.empty_spheres`` on
    this frame) into void regions: spheres of radius at least
    ``probe_radius_ang`` whose probe-centre spheres overlap (module
    docstring). ``grid_spacing_ang`` sets the periodic grid the union
    volumes are measured on and the label grid is exported from;
    ``lining_distance_ang`` is the largest gap between an atom's own sphere
    and a sphere of the region for the atom to line it. Every argument is
    required.
    """
    from scipy.spatial import cKDTree

    frame = _check_frame(frame)
    if not isinstance(spheres, EmptySpheres):
        raise ValueError(f"spheres needs a md_order.EmptySpheres, not "
                         f"{type(spheres).__name__}")
    n = frame.n_atoms
    volume = frame.volume_ang3
    if abs(spheres.box_volume_ang3 - volume) > 1e-9 * volume or (
            len(spheres) and int(spheres.vertices.max()) >= n):
        raise ValueError("the empty spheres were measured on another frame "
                         "(their box volume or atom rows differ from this "
                         "frame's)")
    probe = _nonnegative(probe_radius_ang, "probe_radius_ang")
    h = _positive(grid_spacing_ang, "grid_spacing_ang")
    lining = _nonnegative(lining_distance_ang, "lining_distance_ang")
    radii = {str(k): float(v) for k, v in spheres.radii_ang.items()}
    absent = [s for s in frame.species if s not in radii]
    if absent:
        raise ValueError(f"the spheres carry no radius for {', '.join(absent)}")
    symbols = frame.elements
    box = frame.box_ang
    shape = _grid_shape(box, h)
    n_points = int(np.prod(shape))
    voxel = volume / n_points
    spacing = tuple(float(v) for v in np.linalg.norm(box, axis=1) / shape)
    mean_spacing = (volume / n) ** (1.0 / 3.0)
    tol = md_order.DEGENERATE_REL_TOL * mean_spacing

    radius_all = np.asarray(spheres.radius_ang, dtype=np.float64)
    keep = (radius_all > 0.0) & (radius_all >= probe)
    rows = np.flatnonzero(keep)
    n_left_out = int(radius_all.shape[0] - rows.size)
    n_coincident = 0
    if rows.size:
        cf = np.asarray(spheres.centre_frac, dtype=np.float64)[rows]
        r = radius_all[rows]
        home = bulk._lattice(cf, box)
        idx, image, pos = _images_within(cf, box, tol)
        found = cKDTree(home).sparse_distance_matrix(
            cKDTree(pos), tol, output_type="ndarray")
        i = found["i"]
        j = idx[found["j"]]
        own = (i == j) & ~image[found["j"]].any(axis=1)
        i, j = i[~own], j[~own]
        if i.size:
            group, _, _ = _periodic_components(
                rows.size, i, j, np.zeros((i.size, 3), dtype=np.int64))
            order = np.lexsort((rows, -r))
            first = np.ones(group.max() + 1, dtype=bool)
            representative = np.zeros(rows.size, dtype=bool)
            for s in order.tolist():
                if first[group[s]]:
                    first[group[s]] = False
                    representative[s] = True
            n_coincident = int(rows.size - representative.sum())
            rows = rows[representative]
    base_notes = [
        f"void spheres: empty spheres of radius above 0 and at least the "
        f"probe radius {probe:g} Å ({rows.size} of {radius_all.shape[0]}; "
        f"{n_left_out} left out; {n_coincident} with a centre coinciding with "
        f"another's to within {tol:.3g} Å, DEGENERATE_REL_TOL x the mean "
        "spacing, merged into the larger)",
        f"two voids are connected when their probe-centre spheres overlap: "
        f"d < (r_i - {probe:g}) + (r_j - {probe:g}) Å, with periodic images; "
        "regions are the connected components of that relation",
        f"grid of {shape[0]} x {shape[1]} x {shape[2]} = {n_points} points, "
        f"spacing {spacing[0]:.4g}, {spacing[1]:.4g}, {spacing[2]:.4g} Å along "
        f"a, b, c ({h:g} Å asked for), the point of index 0 at the box corner; "
        f"voxel volume {voxel:.6g} Å^3; the union volume of a region counts "
        "the grid points strictly inside one of its spheres, each point once, "
        "in the sphere it lies deepest inside",
        "radii: " + ", ".join(f"{s} {v:g} Å" for s, v in radii.items())
        + f" ({spheres.radii_source})"]
    empty_int = np.zeros(0, dtype=np.int64)
    empty_float = np.zeros(0)
    if rows.size == 0:
        return VoidRegions(
            probe_radius_ang=probe, grid_spacing_ang=h, spacing_ang=spacing,
            lining_distance_ang=lining, sphere_rows=_frozen(empty_int),
            sphere_region=_frozen(empty_int),
            sphere_cell=_frozen(np.zeros((0, 3), dtype=np.int64)), n_regions=0,
            n_spheres=_frozen(empty_int), volume_sum_ang3=_frozen(empty_float),
            volume_union_ang3=_frozen(empty_float),
            centroid_ang=_frozen(np.zeros((0, 3))),
            elongation=_frozen(empty_float), extent_ang=_frozen(empty_float),
            radius_max_ang=_frozen(empty_float),
            dimensionality=_frozen(empty_int),
            spans=_frozen(np.zeros((0, 3), dtype=bool)),
            copies=_frozen(empty_int),
            label_grid=_frozen(np.full(tuple(shape), -1, dtype=np.int32)),
            region_of_atom=_frozen(np.full(n, -1, dtype=np.int64)),
            lining_atoms=(), void_fraction=0.0,
            percolates=_frozen(np.zeros(3, dtype=bool)), box_ang=box,
            origin_ang=frame.origin_ang, shape=tuple(int(v) for v in shape),
            voxel_volume_ang3=voxel, radii_ang=radii,
            radii_source=spheres.radii_source, n_left_out=n_left_out,
            n_coincident=n_coincident,
            notes=tuple(base_notes + ["no empty sphere reaches the probe "
                                      "radius; there is no void region"]))

    cf = np.asarray(spheres.centre_frac, dtype=np.float64)[rows]
    r = radius_all[rows]
    rp = r - probe
    S = rows.size
    home = bulk._lattice(cf, box)

    # the overlap graph of the probe-centre spheres
    u = v = np.zeros(0, dtype=np.int64)
    shift = np.zeros((0, 3), dtype=np.int64)
    margin = 2.0 * float(rp.max())
    if margin > 0.0:
        idx, image, pos = _images_within(cf, box, margin)
        tree = cKDTree(pos)
        estimate = max(1.0, S / volume * 4.0 / 3.0 * math.pi * margin ** 3)
        per_chunk = int(min(S, max(1, PAIRS_PER_CHUNK // estimate)))
        us, vs, ss = [], [], []
        for start in range(0, S, per_chunk):
            stop = min(S, start + per_chunk)
            found = cKDTree(home[start:stop]).sparse_distance_matrix(
                tree, margin, output_type="ndarray")
            i = found["i"] + start
            j = idx[found["j"]]
            im = image[found["j"]]
            take = (found["v"] < rp[i] + rp[j]) & ~((i == j) & ~im.any(axis=1))
            us.append(i[take])
            vs.append(j[take])
            ss.append(im[take])
        if us:
            u, v, shift, _, _ = _unique_edges(np.concatenate(us),
                                              np.concatenate(vs),
                                              np.concatenate(ss))
    comp, cell, bases = _periodic_components(S, u, v, shift)
    n_regions = len(bases)
    dimensionality, spans, copies = _component_geometry(bases)

    # geometry per region
    vol_s = 4.0 / 3.0 * math.pi * r ** 3
    unwrapped = bulk._lattice(cf + cell, box)
    weight = np.bincount(comp, weights=vol_s, minlength=n_regions)
    n_sph = np.bincount(comp, minlength=n_regions).astype(np.int64)
    s1 = np.column_stack([np.bincount(comp, weights=vol_s * unwrapped[:, a],
                                      minlength=n_regions) for a in range(3)])
    mean = s1 / weight[:, None]
    cov = np.zeros((n_regions, 3, 3))
    for a in range(3):
        for b in range(a, 3):
            cov[:, a, b] = np.bincount(
                comp, weights=vol_s * unwrapped[:, a] * unwrapped[:, b],
                minlength=n_regions) / weight - mean[:, a] * mean[:, b]
            cov[:, b, a] = cov[:, a, b]
    own = np.bincount(comp, weights=vol_s * r ** 2 / 5.0,
                      minlength=n_regions) / weight
    tensors, periods = [], []
    for c in range(n_regions):
        tensor, period = _spanning_tensor(cov[c], bases[c], box,
                                          own[c] * np.eye(3))
        tensors.append(tensor)
        periods.append(period)
    elongation = np.array([_elongation(t) for t in tensors])
    axes = np.array([_principal_axis(t) for t in tensors])
    proj = np.einsum("sk,sk->s", unwrapped, axes[comp])
    finite_extent = (_reduce_by_group(comp, proj + r, n_regions, np.maximum)
                     - _reduce_by_group(comp, proj - r, n_regions, np.minimum))
    extent = np.where(dimensionality == 0, finite_extent,
                      np.array(periods, dtype=np.float64))
    radius_max = _reduce_by_group(comp, r, n_regions, np.maximum)
    home_mean = np.column_stack([
        np.bincount(comp, weights=vol_s * home[:, a], minlength=n_regions)
        for a in range(3)]) / weight[:, None]
    centroid = np.where(dimensionality[:, None] == 0, mean, home_mean)
    first_row = _reduce_by_group(comp, rows, n_regions, np.minimum)

    # the union volume on the grid
    label_flat = np.full(n_points, -1, dtype=np.int32)
    depth_flat = np.full(n_points, -np.inf)
    _mark_spheres(cf, r, comp, box, shape, label_flat, depth_flat)
    del depth_flat
    marked = label_flat >= 0
    union_count = np.bincount(label_flat[marked], minlength=n_regions)
    volume_union = union_count * voxel

    # the lining atoms
    atom_radius = np.array([radii[str(s)] for s in symbols])
    # queried a hair wider than the largest gap kept, so an atom exactly at
    # the limit is found and the gap test below decides
    reach = float(r.max()) + max(radii.values()) + lining + tol \
        + bulk.QUERY_PAD_ANG
    atom_i, _, atom_pos = bulk._image_points(frame, reach)
    atree = cKDTree(atom_pos)
    del atom_pos
    gap_best = np.full(n, np.inf)
    region_of_atom = np.full(n, -1, dtype=np.int64)
    pair_list = []
    estimate = max(1.0, n / volume * 4.0 / 3.0 * math.pi * reach ** 3)
    per_chunk = int(min(S, max(1, PAIRS_PER_CHUNK // estimate)))
    for start in range(0, S, per_chunk):
        stop = min(S, start + per_chunk)
        found = cKDTree(home[start:stop]).sparse_distance_matrix(
            atree, reach, output_type="ndarray")
        s = found["i"] + start
        a = atom_i[found["j"]]
        gap = found["v"] - atom_radius[a] - r[s]
        take = gap <= lining + tol
        s, a, gap = s[take], a[take], gap[take]
        if not s.size:
            continue
        np.minimum.at(gap_best, a, gap)
        best = gap_best[a] == gap
        region_of_atom[a[best]] = comp[s[best]]
        pair_list.append(np.unique(np.column_stack([comp[s], a]), axis=0))
    if pair_list:
        pairs = np.unique(np.concatenate(pair_list), axis=0)
    else:
        pairs = np.zeros((0, 2), dtype=np.int64)

    # regions largest first
    rank = np.lexsort((first_row, -weight, -union_count))
    relabel = np.empty(n_regions, dtype=np.int64)
    relabel[rank] = np.arange(n_regions)
    label_flat[marked] = relabel[label_flat[marked]]
    region_of_atom = np.where(region_of_atom >= 0,
                              relabel[np.maximum(region_of_atom, 0)], -1)
    lining_rows = [np.zeros(0, dtype=np.int64) for _ in range(n_regions)]
    if pairs.shape[0]:
        order = np.argsort(pairs[:, 0], kind="stable")
        sorted_pairs = pairs[order]
        bounds = np.searchsorted(sorted_pairs[:, 0], np.arange(n_regions + 1))
        for c in range(n_regions):
            lining_rows[relabel[c]] = sorted_pairs[bounds[c]:bounds[c + 1], 1]
    lining_atoms = tuple(_frozen(np.sort(a)) for a in lining_rows)
    percolates = spans.any(axis=0)
    void_fraction = float(marked.sum()) / n_points

    notes = base_notes + [
        f"{n_regions} void region(s) from {S} spheres; union volumes sum to "
        f"{float(volume_union.sum()):.6g} Å^3 ({void_fraction:.6g} of the box; "
        f"summed sphere volumes {float(weight.sum()):.6g} Å^3); the largest "
        f"holds {int(n_sph[rank][0])} sphere(s), {volume_union[rank][0]:.6g} "
        f"Å^3; regions spanning an axis: {int(spans.any(axis=1).sum())} (axes "
        f"spanned by any: {_spans_text(percolates)})",
        "elongation: sqrt of the largest over the smallest eigenvalue of the "
        "volume-weighted gyration tensor of the sphere centres plus each "
        "sphere's own r^2 / 5; a finite region is unwrapped into its one "
        "connected copy and its extent runs along the largest eigenvector "
        "from sphere surface to sphere surface; a region spanning one axis "
        "takes the spread perpendicular to its translation T and |T|^2 / 12 "
        "along it, its extent is |T| (one period) and its centroid is that of "
        "its spheres as wrapped into the box; a region spanning two or three "
        "axes has no elongation and no extent (NaN)",
        f"lining atoms: gap between the atom's sphere and a sphere of the "
        f"region at most {lining:g} Å (lining_distance_ang; to {tol:.3g} Å); "
        f"{int((region_of_atom >= 0).sum())} of {n} atoms line a region"]
    split = np.flatnonzero(copies[rank] > 1)
    if split.size:
        notes.append(
            f"{split.size} region(s) stand for several interpenetrating "
            "pieces that a box translation maps onto each other (copies > 1): "
            + ", ".join(f"region {c} x{int(copies[rank][c])}" for c in split[:10]))
    return VoidRegions(
        probe_radius_ang=probe, grid_spacing_ang=h, spacing_ang=spacing,
        lining_distance_ang=lining, sphere_rows=_frozen(rows),
        sphere_region=_frozen(relabel[comp]), sphere_cell=_frozen(cell),
        n_regions=n_regions, n_spheres=_frozen(n_sph[rank]),
        volume_sum_ang3=_frozen(weight[rank]),
        volume_union_ang3=_frozen(volume_union[rank]),
        centroid_ang=_frozen(frame.origin_ang + centroid[rank]),
        elongation=_frozen(elongation[rank]), extent_ang=_frozen(extent[rank]),
        radius_max_ang=_frozen(radius_max[rank]),
        dimensionality=_frozen(dimensionality[rank]), spans=_frozen(spans[rank]),
        copies=_frozen(copies[rank]),
        label_grid=_frozen(label_flat.reshape(tuple(shape))),
        region_of_atom=_frozen(region_of_atom), lining_atoms=lining_atoms,
        void_fraction=void_fraction, percolates=_frozen(percolates),
        box_ang=box, origin_ang=frame.origin_ang,
        shape=tuple(int(x) for x in shape), voxel_volume_ang3=voxel,
        radii_ang=radii, radii_source=spheres.radii_source,
        n_left_out=n_left_out, n_coincident=n_coincident, notes=tuple(notes))


def select_regions(regions, *, min_volume_ang3: float, min_elongation: float
                   ) -> np.ndarray:
    """The region indices with ``volume_ang3 >= min_volume_ang3`` and
    ``elongation >= min_elongation``, on an :class:`AccessibleRegions` or a
    :class:`VoidRegions`. Both thresholds are required; 0 and 0 select every
    region with a defined elongation."""
    if not hasattr(regions, "volume_ang3") or not hasattr(regions, "elongation"):
        raise ValueError("accessible_regions or void_regions results are "
                         "needed")
    volume = _nonnegative(min_volume_ang3, "min_volume_ang3")
    elongation = _nonnegative(min_elongation, "min_elongation")
    with np.errstate(invalid="ignore"):
        mask = (np.asarray(regions.volume_ang3) >= volume) \
            & (np.asarray(regions.elongation) >= elongation)
    return np.flatnonzero(mask)


# ---------------------------------------------------------------------------
# frame averages through md_stats
# ---------------------------------------------------------------------------

def _landscape_setting(landscape: Landscape) -> tuple:
    return tuple(sorted((k, str(v)) for k, v in
                        landscape.method_parameters.items()))


def average_accessible_fraction(landscapes: Sequence[Landscape], deltas_vu, *,
                                frames: Sequence[int] | None = None) -> Series:
    """The accessible volume fraction at each Delta of ``deltas_vu``
    (strictly increasing, in valence units), per frame and averaged over
    frames as a :class:`facet.core.md_stats.Series` on the Delta axis. The
    frames must share the probe, grid spacing, r_cut, parameter set and
    repulsion."""
    what = "average_accessible_fraction"
    results = _results(landscapes, Landscape, what)
    _one_setting(results, what, **{"landscape settings": _landscape_setting})
    deltas = np.asarray(deltas_vu, dtype=np.float64)
    if deltas.ndim != 1 or deltas.size == 0:
        raise ValueError(f"{what}: deltas_vu needs a 1-D array of thresholds")
    rows = [accessible_fraction(r, deltas) for r in results]
    first = results[0]
    notes = (f"accessible volume fraction: non-excluded grid points with "
             f"|V - {first.target_vu:g}| <= Delta over all grid points; "
             + "; ".join(f"{k} = {v}" for k, v in
                         first.method_parameters.items()),)
    return Series.from_frames(
        deltas, rows, name=f"accessible volume fraction of {first.probe_label}",
        axis_name="delta_vu", axis_unit="v.u.", value_unit="1", frames=frames,
        notes=notes)


def average_percolation_thresholds(results: Sequence[PercolationThresholds], *,
                                   frames: Sequence[int] | None = None
                                   ) -> dict[str, Scalar]:
    """The percolation threshold along a, b and c, and the smallest of the
    three (``'any'``), each a :class:`facet.core.md_stats.Scalar` over
    frames. A frame whose region never spans an axis contributes NaN there,
    which the Scalar counts and notes."""
    what = "average_percolation_thresholds"
    results = _results(results, PercolationThresholds, what)
    _one_setting(results, what, **{"landscape settings": lambda r: tuple(
        sorted((k, str(v)) for k, v in r.landscape_parameters.items()))})
    first = results[0]
    label = _ox_label(first.probe, first.probe_ox)
    notes = (f"the smallest Delta at which an accessible region of {label} "
             "spans the axis (bisection over the grid's mismatch values); "
             + "; ".join(f"{k} = {v}" for k, v in
                         first.landscape_parameters.items()),)
    out = {}
    for k, axis in enumerate(AXES):
        out[axis] = Scalar(f"percolation threshold of {label} along {axis}",
                           "v.u.", [float(r.delta_vu[k]) for r in results],
                           frames=frames, notes=notes)
    out["any"] = Scalar(f"percolation threshold of {label}, any axis", "v.u.",
                        [r.delta_any_vu for r in results], frames=frames,
                        notes=notes)
    return out


def average_modifier_density(results: Sequence[ModifierDensity], *,
                             frames: Sequence[int] | None = None) -> dict:
    """The modifier count per anion (md_stats Distribution, fractions over
    the counts 0 .. the largest seen), the modifier-rich fraction, the
    number of clusters, the largest cluster's shares, the fraction of
    modifiers in a cluster, whether the clusters span each axis (Scalars,
    0 or 1 per frame), and, when every frame carries a speciation, the
    modifier-rich anions by speciation category (Distribution) and the rich
    fraction within each category (Scalars)."""
    what = "average_modifier_density"
    results = _results(results, ModifierDensity, what)
    _one_setting(results, what, **{
        "modifiers": lambda r: tuple(sorted(r.modifiers)),
        "anions": lambda r: tuple(sorted(r.anions)),
        "k_rich": lambda r: r.k_rich,
        "cutoffs (Å)": lambda r: tuple(sorted(r.cutoffs_ang.items()))})
    first = results[0]
    mods = ", ".join(sorted(first.modifiers))
    ans = ", ".join(sorted(first.anions))
    notes = (f"modifier count: {mods} atoms within the cutoff of each {ans} "
             "atom (" + "; ".join(
                 f"{a}-{b} {v:g} Å, {first.cutoff_sources[(a, b)]}"
                 for (a, b), v in sorted(first.cutoffs_ang.items()))
             + f"); modifier-rich at {first.k_rich} or more (k_rich)",)
    top = max(max(r.count_distribution()) for r in results)
    out = {
        "count": Distribution.from_counts(
            [r.count_distribution() for r in results],
            name=f"modifier count per anion ({mods} around {ans})",
            kind="fraction", keys=tuple(range(top + 1)), frames=frames,
            notes=notes),
        "rich fraction": Scalar(
            f"modifier-rich anion fraction (k_rich {first.k_rich})", "1",
            [r.fraction_rich for r in results], frames=frames, notes=notes),
        "clusters": Scalar("number of modifier clusters", "1",
                           [float(r.n_clusters) for r in results],
                           frames=frames, notes=notes),
        "largest cluster fraction": Scalar(
            "largest modifier cluster, fraction of cluster nodes", "1",
            [r.largest_cluster_fraction for r in results], frames=frames,
            notes=notes),
        "largest cluster rich fraction": Scalar(
            "largest modifier cluster, fraction of the modifier-rich anions",
            "1", [r.largest_cluster_rich_fraction for r in results],
            frames=frames, notes=notes),
        "modifiers in clusters": Scalar(
            "fraction of modifier atoms in a cluster", "1",
            [r.modifiers_in_clusters / r.n_modifiers if r.n_modifiers else
             math.nan for r in results], frames=frames, notes=notes),
    }
    for k, axis in enumerate(AXES):
        out[f"spans {axis}"] = Scalar(
            f"modifier clusters span {axis} (1 yes, 0 no)", "1",
            [float(r.percolates[k]) for r in results], frames=frames,
            notes=notes)
    with_speciation = [r.speciation is not None for r in results]
    if any(with_speciation) and not all(with_speciation):
        raise ValueError(f"{what}: the frames were measured with different "
                         "speciation inputs (bonds and formers for some "
                         "frames and not for others); an average over frames "
                         "needs one")
    if all(with_speciation):
        _one_setting(results, what, **{
            "formers": lambda r: tuple(sorted(r.formers)),
            "v_bond (v.u.)": lambda r: r.v_bond_vu})
        spec_notes = notes + (
            f"speciation by bonded formers ({', '.join(sorted(first.formers))})"
            f" at v_bond {first.v_bond_vu:g} v.u.: free 0, NBO 1, BO 2, "
            "tricluster 3 or more",)
        out["rich by speciation"] = Distribution.from_counts(
            [{key: r.speciation[key][0] for key in glass.SPECIATION_KEYS}
             for r in results], name="modifier-rich anions by speciation",
            kind="fraction", keys=glass.SPECIATION_KEYS, frames=frames,
            notes=spec_notes)
        for key in glass.SPECIATION_KEYS:
            out[f"rich fraction among {key}"] = Scalar(
                f"modifier-rich fraction among {key} anions", "1",
                [r.speciation[key][0] / r.speciation[key][1]
                 if r.speciation[key][1] else math.nan for r in results],
                frames=frames, notes=spec_notes)
    return out


def average_void_regions(results: Sequence[VoidRegions], *, volume_edges_ang3,
                         elongation_edges, frames: Sequence[int] | None = None
                         ) -> dict:
    """The void fraction, the number of regions, the largest region's union
    volume and the number of spanning regions (Scalars), and the region
    union volumes and elongations binned on the edges given (Histograms),
    over frames. The frames must share the probe radius, grid spacing,
    lining distance and radii."""
    what = "average_void_regions"
    results = _results(results, VoidRegions, what)
    _one_setting(results, what, **{
        "probe radii (Å)": lambda r: r.probe_radius_ang,
        "grid spacings asked for (Å)": lambda r: r.grid_spacing_ang,
        "lining distances (Å)": lambda r: r.lining_distance_ang,
        "radii (Å)": lambda r: tuple(sorted(r.radii_ang.items())),
        "radii source": lambda r: r.radii_source})
    first = results[0]
    notes = (f"probe radius {first.probe_radius_ang:g} Å; grid spacing "
             f"{first.grid_spacing_ang:g} Å asked for; lining distance "
             f"{first.lining_distance_ang:g} Å; radii: " + ", ".join(
                 f"{s} {v:g} Å" for s, v in first.radii_ang.items())
             + f" ({first.radii_source})",)
    return {
        "void fraction": Scalar("void fraction (union of the void spheres)",
                                "1", [r.void_fraction for r in results],
                                frames=frames, notes=notes),
        "regions": Scalar("number of void regions", "1",
                          [float(r.n_regions) for r in results], frames=frames,
                          notes=notes),
        "largest region volume": Scalar(
            "largest void region, union volume", "Å^3",
            [float(r.volume_union_ang3[0]) if r.n_regions else 0.0
             for r in results], frames=frames, notes=notes),
        "spanning regions": Scalar(
            "void regions spanning an axis", "1",
            [float(r.spans.any(axis=1).sum()) if r.n_regions else 0.0
             for r in results], frames=frames, notes=notes),
        "region volume": Histogram.from_samples(
            [r.volume_union_ang3 for r in results], volume_edges_ang3,
            name="void region union volume", unit="Å^3", frames=frames,
            notes=notes),
        "region elongation": Histogram.from_samples(
            [r.elongation for r in results], elongation_edges,
            name="void region elongation", unit="1", frames=frames,
            notes=notes),
    }
