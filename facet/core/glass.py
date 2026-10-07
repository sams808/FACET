"""Glass descriptors of an MD model, per frame and averaged over frames.

What a glass paper reports about a model is a set of distributions: the
coordination numbers of each element, the Q^n speciation of the network
formers, the fraction of four-coordinated boron, the bridging and
non-bridging oxygens, the bond-angle and bond-length distributions, the
partial pair distribution functions. This module measures each of them on
every frame of a trajectory and averages them with their spread
(:mod:`facet.core.md_stats`), from the arrays of the bulk engine
(:mod:`facet.core.bulk`). It measures; it does not run, build or edit MD.

ONE BOND DEFINITION, AND A SECOND ONE BESIDE IT
-----------------------------------------------
FACET cuts a bond by partial bond valence [1]: a cation-anion contact is a
bond when its valence exceeds ``v_bond``. Every bond-based descriptor here (the CN,
the oxygen speciation, Q^n, Q^n(mX), the linkages, the halide environments,
the angles, the bond lengths) is computed from one list of bonds,
``bulk.bonds_at(table, v_bond)``, the same list the CN counts, so the
descriptors can never disagree with each other about which atoms are bonded.

Glass work usually cuts bonds by distance instead, at the first minimum of the
partial g(r). The same descriptors are therefore computed a second time from
:class:`DistanceBonds`, the cation-anion pairs closer than a cutoff, and the
two are reported side by side, with the per-atom cross-table of the two CN
(``GlassResult.comparison``). Where the two definitions give different
numbers on the same atoms, that difference is the measurement FACET exists to
show. The distance cutoffs are never typed in: each one is the first
minimum of the frame-averaged partial g(r) of the model, found by a stated
rule, or a value the user gives, recorded with its source.

PARTIAL PAIR DISTRIBUTION FUNCTIONS
-----------------------------------
:func:`partial_rdf` takes the pair blocks of ``bulk.iter_pairs``, the one
pair search of a frame, and never searches again. With N_a atoms of element
a, N_b of b and the box volume V, the radial distribution per a atom is
deposited on pdf.py's grid ``r_k = arange(dr, r_max + dr/2, dr)``
(``pdf.py:440``) with ``pdf._deposit`` (``pdf.py:526-546``), which spreads each
distance linearly over its two neighbouring grid points so a peak sits at its
distance, and normalised as pdf.py normalises g(r) (``pdf.py:504-520``)::

    g_ab(r_k) = deposit_ab(r_k) V / (4 pi r_k^2 dr N_a N_b)

which is R_ab / (4 pi r^2 rho_b) with rho_b = N_b / V, the convention of
Keen [2] for a partial; each unordered pair is deposited once, so g_ab and
g_ba are the same array, and for a = b both orders count, so g_aa tends to
1 - 1/N_a at large r in a model of N_a atoms. The running coordination number
N_ab(r_k), the mean number of b atoms within r_k of an a atom, is counted, not
integrated: a pair counts at every grid point r_k >= d, exactly, so N_ab at a
grid point equals the mean distance-cut CN at that cutoff
(``tests/test_glass.py`` checks both against ``bulk.distance_cn``).

The grid stops at half the smallest perpendicular width of the box (noted when
the request was larger). Beyond it, a sphere around an atom overlaps its own
periodic images and the shell is no longer inside one box; the minimum-image
convention ends there [3]. The pair search itself goes one grid step further,
so the last grid point receives every share ``_deposit`` gives it.

THE FIRST MINIMUM
-----------------
:func:`first_minimum` locates the minimum by one of two rules, chosen by the
caller in a :class:`MinimumMethod` with no default. Both compare g with its
uncorrelated level L: 1 for a pair of two elements, and 1 - 1/N_a for a pair
of one element, the large-r limit of g_aa in this normalisation (each of the
N_a atoms has N_a - 1 partners of its own element), taken from the model's own
atom count:

* ``'valley'``: the lowest point of g(r) between the first maximum, once g
  has risen above L, and the place where g, having fallen below L, next rises
  above it;
* ``'first local minimum'``: the first local minimum after the first local
  maximum above L, the textbook reading.

A g(r) measured on a finite model carries counting noise. On a 3 000-atom
Na2O-3SiO2 glass (Pedone potential, LAMMPS, 20 frames at 300 K) the
frame-averaged g_NaO dips below 1 at 2.77-2.78 Å (0.987 and 0.996, between
1.024 and 1.018), where its counting error is 0.05, and comparing g with 1 as
it stands put the Na-O cutoff on that dip (N_NaO 3.75) rather than in the
valley floor at 3.0-3.4 Å (N_NaO about 5.5; LAMMPS ``compute rdf`` puts its
minimum at 3.23 Å). On single frames of the four melt-quench models the same
comparison put Na-O at 2.14-2.77 Å, from the rising edge of the first peak
on, and Si-O as low as 1.71 Å, in the tail of the Si-O bond lengths
(verification by three independent checks, 2026-10-06). Every comparison is
therefore made against the counting error of g, with a margin the caller
states (``margin_std_errors``, a method choice with no default; 0 compares
the values as they stand, which is the rule on a noise-free g):

* the standard error of g at a grid point is Poisson's, from the number of
  pairs deposited there over the frames averaged: g / sqrt(n_pairs), with
  n_pairs = g x :func:`pair_counts_per_unit_g` (n_frames 4 pi r^2 dr N_a N_b
  / V, halved for a pair of one element, whose pairs are deposited from both
  ends). A deposit is a sum of hat weights of at most 1, so its Poisson
  variance, the sum of the squared weights, is at most the deposit itself:
  the error used is that upper bound. For a box that changes between frames V
  is the mean volume of the frames used, exact for the error at the level L
  and approximate for the observed one;
* g is above L at a grid point only where it exceeds L by more than the
  margin times the larger of two errors: g's own, and the error g would have
  at L. g's own keeps one or two pairs at an unusually short distance from
  reading as a peak (a point needs more than about (margin)^2 pairs);
* g is below L over a stretch of consecutive grid points where it lies below
  L, and only when the pairs that stretch holds fall short of the number
  expected at L by more than the margin times the Poisson error of the
  larger of the two numbers (a pair's hat weights inside a stretch sum to at
  most 1, so the square root of the summed count bounds that error). A gap
  is judged by all the pairs it lacks: in a small crystal cell fewer pairs
  than (margin)^2 may be expected at L per grid point, and a point-by-point
  test then misses whole gaps (cryolite 2 x 2 x 2, one frame: no Al-F
  minimum, and Na-F at 3.6 Å past the 2.82-3.24 Å gap), where judged as a
  stretch it gives Al-F in the 1.81-3.97 Å gap and Na-F at 2.46 Å, in the
  2.35-2.57 Å gap. A stretch drawn by noise just below L, if it passes, only
  starts the search window early; the window's lowest point decides;
* a local maximum ends, and a local minimum ends, only at a drop, or a rise,
  larger than the margin times the combined error of the two points;
* the floor of the valley is every grid point within the margin times the
  combined error of the lowest point, from the first such point to the last,
  so a fragmented floor (zeros between single pairs in the tail of a Si-O
  peak) is one floor; ``flat_rule`` takes its first point or its midpoint.
  The valley search starts after the last point above L before g first falls
  below it, so a zero between two distances of one crystal shell is not a
  valley unless it is below L by the margin (cryolite: the 2.227 and 2.263 Å
  Na-F distances, split at 2.24 Å when compared as they stand).

Measured with margin 2 and no smoothing on the four models (20 or 10 frames
averaged, cation-anion pairs), the valley rule gives Si-O 2.19-2.29 Å
(LAMMPS 1.99-2.09), B-O 2.03 (1.95), Al-O 2.44 (2.41) and Na-O 3.04-3.26 Å
(2.91-3.23), each LAMMPS point inside the floor FACET reports; on single
frames, Na-O 3.12-3.52 Å and Si-O 2.28-2.45 Å. Compared as they stand
(margin 0), single frames give Na-O 2.14-2.77 Å and Si-O down to 1.71 Å, in
the tail of the Si-O bond lengths. On a g(r) with a valley floor of 0.6,
Poisson-sampled at those models' pair counts, 1 of 200 seeds (20 frames) and
4 of 200 (one frame) put the valley-rule point where the noise-free g is
above 0.8, each beside a count 3.1-4.4 standard errors from its noise-free
value, against 181 and 198 of 200 compared as they stand. The 'first local
minimum' rule follows the local shape of g and stays sensitive to noise
without smoothing (NS3 Na-O on one frame 2.34-3.00 Å at margin 2; 134 of the
200 sampled g at 20 frames above 0.8); smoothed by 0.03 Å and judged at
margin 2 it put 0 of 200 (20 frames) and 4 of 200 (one frame) above 0.8. A
note is attached when the lowest point it finds lies above L.
``tests/test_glass.py`` pins the rules on that sampled g(r) and on a
jittered quartz model, where comparing as they stand lands on the noise.

Each result carries its floor (``floor_r_ang``), and
:func:`analyse_trajectory` states with every automatic cutoff the range of
N_ab(r) across that floor, which is the spread a CN inherits from where in
the floor the cutoff sits. Smoothing is optional and named
(``smooth_sigma_ang``, a Gaussian truncated at
:data:`SMOOTH_HALF_WIDTH_SIGMAS`, pdf._broaden's truncation, normalised at the
grid ends so the ends are not pulled towards 0); the errors are smoothed with
it (the variance by the squared kernel). A kernel wider than the g(r) grid
the arguments ask for is refused before any frame is read; one wider than
the grid left after the cap at half the box width gives that pair no cutoff,
with the reason. ``RdfMinimum.g_value`` and ``g_std_error`` are g as
measured at the point and its counting error, whatever the smoothing; the
floor and the depth are read on the g the rule reads. The located point is a
grid point, so the distance-cut CN at it equals N_ab(r) there. A pair whose
g(r) has no such point (a dilute pair, a molecule in a large box) gets no
cutoff, with the reason, and the distance descriptors that need it are not
reported, again with the reason. A user cutoff overrides any rule, and its
entry in ``GlassResult.minima`` says so.

TWO PASSES, AND WHEN THERE IS ONE
---------------------------------
The automatic cutoffs come from the frame-averaged g(r), which exists only
after every frame has been searched. :func:`analyse_trajectory` therefore runs
pass 1 (one ``bulk.iter_pairs`` per frame feeding the valence table, the
partial g(r) and, when the user gave every cation-anion cutoff, the distance
bonds), and then, only when some cutoff is automatic, pass 2: each frame read
again and searched once more with ``bulk.iter_pairs`` at the largest cutoff,
about half the bond-valence radius. Giving ``cutoffs_ang`` (for example
``GlassResult.cutoffs_ang`` of a previous run) makes it one pass.

NETWORK FORMERS AND WHAT IS COUNTED
-----------------------------------
Which cations form the network is chemistry, so there is no default
(:data:`FORMERS_DEFAULT` is None): every former-dependent function takes the
set explicitly, and ``analyse_trajectory(formers=None)`` reports none of them,
with a note. A former is a cation of the model (oxidation state >= 0). With a
former set F, after Zachariasen's network picture [4]:

* an anion bonded to 0 formers is ``'free'``, to exactly 1 ``'NBO'``, to
  exactly 2 ``'BO'``, to 3 or more ``'tricluster'``, the four categories of the
  MD prompt made disjoint so they sum to 1; "bridging" in the Q^n sense below
  is BO + tricluster;
* Q^n of a former: n = the number of its bonded anions that are bridging
  (bonded to two or more formers), the notation of Lippmaa et al. [5];
* Q^n(mX): of those n bridges, m = the number of T-O-X linkages to formers of
  element X, counted over the other formers on each bridging anion (with a
  tricluster, one anion links to two). Without triclusters the m over every X
  sum to n. Labelled ``'Q4(2Al)'``, the Si(nAl) reading of 29Si NMR;
* network connectivity: the mean n over the atoms of a former element, and
  over every former. Hill [6] defines a connectivity from the composition;
  this one is measured on the model's bonds;
* linkages X-O-Y: per anion, every pair of its bonded cations, formers or not.
  Al-O-Al is the linkage Loewenstein's rule excludes between tetrahedra [7];
  it is counted here whatever the Al coordination;
* N4: boron with CN 3, 4 and any other CN, which is reported, never folded in
  (Yun and Bray read N4 from 11B NMR [8]); Al with CN 4, 5, 6 and other;
* for phosphorus, Q^n counts bridging anions by their bonds, and a terminal
  oxygen of a P=O double bond is an NBO like any other: the bond-valence and
  distance definitions do not separate double from single bonds. Brow
  reviews the Q^n description of phosphate glasses and their terminal P=O
  oxygen [9]; no model of the double bond is applied here;
* anion environments: per anion, its bonded cations by element, labelled
  ``'F-Al1Na2'`` (elements in alphabetical order), for any anion;
* phi and the plateau width of every element (the lone-pair link to the
  crystal work), CN of every element by both definitions.

Bond angles are atan2(|u x v|, u . v), not acos of the cosine, which loses
half its digits near 0 and 180 degrees (``planes.py:368-374``). Bond lengths
are the bonds' own distances.

Physically impossible values are not refused when they depend on the
threshold (a slider must be able to cross them, Step 0 reply, section 4 item
11); they become notes with the threshold: an anion bonded to no cation, and
Si with CN above 6 (:data:`CN_NOTE_LIMITS`, the example the MD prompt names).
Fraction sets are checked by ``md_stats.check_fractions``.

COMPOSITION
-----------
:func:`composition` counts atoms, gives atomic per cent, the net charge from
the oxidation states as an exact integer, the mass from gemmi's standard
atomic weights, the oxide mol % for an oxide basis the user gives (exact
rational arithmetic, so a balanced basis leaves exactly 0 anions unassigned),
and :func:`density_g_per_cm3` the density, with the atomic mass constant from
scipy.constants. A non-neutral model is a note, as ``Structure.net_charge``
reports it on the crystal side.

TIMINGS
-------
Measured 2026-10-06 on this machine (Windows 11, i5-13420H, Python 3.11.9,
numpy 2.4.6, scipy 1.15.1), unpinned, frames held in memory (no file is
parsed), on two boxes: a 10 x 10 x 10 quartz supercell (9 000 atoms, formers
{'Si'}) and ``tools/bench_md.make_box(21)`` (9 261 atoms, Si/O/Na at random:
the chemistry is meaningless, only the size matters). ``analyse_trajectory``
end to end, median [min-max] of 3 runs after one untimed run, quartz /
random box. The machine was saturated throughout (Win32_Processor
LoadPercentage 100) by an 8-thread LAMMPS run and benchmarks of other
sessions, and ``bulk.analyse_frame`` alone took 2.1 / 1.7 s, against the
0.55-1.1 s bulk.py records unpinned; read every figure as 2-3 times what a
quiet machine gives::

    g(r) to   pass 1, per frame   pass 2, per frame   1 frame end to end
    6 Å       3.7 / 3.2 s         0.63 / 0.60 s       4.5 [3.9-4.7] / 3.9 [3.8-4.4] s
    10 Å      15.3 / 10.5 s       0.97 / 1.00 s       16.4 [15.5-16.5] / 11.6 [10.7-12.4] s

Three frames at 10 Å in one pass (every cutoff given) took 29.7 [27.7-33.8]
/ 28.0 [24.9-32.0] s, 8.2-12.0 s per frame. The tracemalloc peak for one
frame at 10 Å was 119 / 120 MB. The time follows the g(r) radius, because the
one pair search per frame goes out to it: under cProfile, in an earlier round
on a less loaded machine (pass 1 at 10 Å then took 5.5-7.9 s per frame), the
search, ``bulk.iter_pairs`` with the vector of every pair, was 58 % of the
run, the g(r) deposition 18 % and the descriptors of both definitions 14 %.
Since then the angle and bond-length grouping uses integer codes in place of
a Python set over every angle, and the g(r) grid index is computed rather
than searched (:func:`_first_at_or_above`). :func:`first_minimum` with its
counting errors takes 0.15-0.64 ms on the 1 200-point (to 12 Å) 20-frame
NS3 g_NaO (either rule, unsmoothed or smoothed by 0.03 Å; minimum of 5 x 20
calls, two runs, 2026-10-07, machine loaded by other sessions), so a minimum
per element pair is negligible beside the pair search. Automatic cutoffs read against the
counting error lie further out than the noise crossings they replace (Na-O
3.26 against 2.77 Å on the NS3 model), so pass 2 searches a little further.
Memory does not grow with the number of frames beyond the per-frame rows of
each result (a few thousand numbers per descriptor) and one int32 CN per
atom per frame, kept for the cross-table of pass 2.

REFERENCES
----------
[1] I. D. Brown, "Recent developments in the methods and applications of the
    bond valence model", *Chemical Reviews* 109 (2009) 6858-6919,
    https://doi.org/10.1021/cr900053k
[2] D. A. Keen, "A comparison of various commonly used correlation functions
    for describing total scattering", *Journal of Applied Crystallography* 34
    (2001) 172-177, https://doi.org/10.1107/S0021889800019993
[3] M. P. Allen and D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed.,
    Oxford University Press, Oxford (2017).
[4] W. H. Zachariasen, "The atomic arrangement in glass", *Journal of the
    American Chemical Society* 54 (1932) 3841-3851,
    https://doi.org/10.1021/ja01349a006
[5] E. Lippmaa, M. Maegi, A. Samoson, G. Engelhardt and A.-R. Grimmer,
    "Structural studies of silicates by solid-state high-resolution 29Si
    NMR", *Journal of the American Chemical Society* 102 (1980) 4889-4893,
    https://doi.org/10.1021/ja00535a008
[6] R. Hill, "An alternative view of the degradation of bioglass", *Journal
    of Materials Science Letters* 15 (1996) 1122-1125,
    https://doi.org/10.1007/BF00539955
[7] W. Loewenstein, "The distribution of aluminum in the tetrahedra of
    silicates and aluminates", *American Mineralogist* 39 (1954) 92-96,
    http://www.minsocam.org/ammin/AM39/AM39_92.pdf
[8] Y. H. Yun and P. J. Bray, "Nuclear magnetic resonance studies of the
    glasses in the system Na2O-B2O3-SiO2", *Journal of Non-Crystalline
    Solids* 27 (1978) 363-380, https://doi.org/10.1016/0022-3093(78)90020-0
[9] R. K. Brow, "Review: the structure of simple phosphate glasses",
    *Journal of Non-Crystalline Solids* 263-264 (2000) 1-28,
    https://doi.org/10.1016/S0022-3093(99)00620-1
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, \
    Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from . import bulk, bv
from .md_model import Frame, FrameError, ModelOxidation, Trajectory, \
    validate_symbol
from .md_stats import Distribution, Histogram, Provenance, Scalar, Series, \
    _bin, _compact, check_fractions
from .pdf import _deposit

__all__ = [
    "FORMERS_DEFAULT", "RDF_DR_ANG", "MINIMUM_RULES", "FLAT_RULES",
    "SMOOTH_HALF_WIDTH_SIGMAS", "SPECIATION_KEYS", "CN_NOTE_LIMITS",
    "DEFINITIONS",
    "AnalysisCancelled", "MinimumMethod", "RdfMinimum", "PartialRDF",
    "HistogramBins", "DistanceBonds", "ModelComposition", "GlassResult",
    "rdf_grid_ang", "partial_rdf", "pair_counts_per_unit_g", "first_minimum",
    "distance_bonds",
    "bond_cn", "cn_counts", "cn_groups", "boron_n4", "aluminium_cn",
    "former_bond_counts", "anion_speciation", "qn_counts", "qn_mx_counts",
    "connectivity", "linkage_counts", "anion_environments",
    "lone_pair_values", "bond_angles_deg", "bond_lengths_ang",
    "composition", "density_g_per_cm3", "analyse_trajectory",
]

# ---------------------------------------------------------------------------
# constants: each one is a choice or a definition, and says which
# ---------------------------------------------------------------------------

# No network-former set is assumed anywhere. TODO(Sam): the UI default (Step 0
# reply, section 6); every function that needs formers takes them explicitly.
FORMERS_DEFAULT: frozenset[str] | None = None

# The r grid step of the partial g(r): pdf.pair_distribution's default
# (pdf.py:412). A grid choice, stated in every result and overridable.
RDF_DR_ANG: float = 0.01

MINIMUM_RULES = ("valley", "first local minimum")
FLAT_RULES = ("first", "midpoint")

# The optional smoothing Gaussian is cut at this many sigma, as pdf._broaden
# cuts its peaks (pdf.py:552); the weight left out is 2e-9 of the whole.
SMOOTH_HALF_WIDTH_SIGMAS: float = 6.0

# The anion speciation categories, disjoint so they sum to 1: 0, 1, 2 and 3 or
# more bonded formers.
SPECIATION_KEYS = ("free", "NBO", "BO", "tricluster")

# CN values the MD prompt names as physically impossible ("Si CN > 6"). They
# become a note with the threshold, never a refusal (a CN depends on v_bond).
# No other element has a limit here: none was given.
CN_NOTE_LIMITS: dict[str, int] = {"Si": 6}

DEFINITIONS = ("bv", "distance")

# atan2(|u x v|, u . v) lies in [0, 180] degrees, and phi = |sum v u| / sum v
# in [0, 1] (the triangle inequality): the natural ranges of the two
# histograms whose edges do not depend on the data.
_ANGLE_RANGE_DEG = 180.0
_PHI_RANGE = 1.0


class AnalysisCancelled(Exception):
    """The caller's ``cancelled()`` hook returned True before any frame was done."""


# ---------------------------------------------------------------------------
# argument checks
# ---------------------------------------------------------------------------

def _positive(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} {value!r} is not a number")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} {value!r} is not a number") from None
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} is {value!r}; a finite number above 0 is "
                         "needed")
    return out


def _threshold(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} {value!r} is not a number")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} {value!r} is not a number") from None
    if not math.isfinite(out):
        raise ValueError(f"{name} is {value!r}; a finite valence is needed")
    return out


def _symbols(elements) -> np.ndarray:
    """The (N,) symbol array of a Frame, or of an array of symbols."""
    if isinstance(elements, Frame):
        return elements.elements
    out = np.asarray(elements)
    if out.ndim != 1 or out.dtype.kind != "U":
        raise ValueError("elements needs one element symbol per atom (a "
                         "Frame's elements, for instance)")
    return out


def _symbol_set(values, what: str) -> frozenset[str]:
    if values is None:
        raise ValueError(f"{what}: no set given")
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{what} needs a collection of element symbols, not "
                         "one string")
    try:
        items = list(values)
    except TypeError:
        raise ValueError(f"{what} needs a collection of element symbols") \
            from None
    return frozenset(validate_symbol(str(v)) for v in items)


def _formers(formers) -> frozenset[str]:
    """The former set, which every former-dependent function requires."""
    if formers is None:
        raise ValueError(
            "network formers: none given. No former set is assumed "
            "(FORMERS_DEFAULT is None); pass the formers explicitly, for "
            "example {'Si', 'Al', 'B'}")
    return _symbol_set(formers, "formers")


def _bond_arrays(bonds, n_atoms: int):
    """(cation, anion, vec_ang, d_ang) of a bulk.Bonds or a DistanceBonds."""
    try:
        cation = np.asarray(bonds.cation)
        anion = np.asarray(bonds.anion)
        vec_ang = np.asarray(bonds.vec_ang, dtype=np.float64)
        d_ang = np.asarray(bonds.d_ang, dtype=np.float64)
    except AttributeError:
        raise ValueError("bonds needs cation, anion, vec_ang and d_ang arrays "
                         "(bulk.bonds_at and distance_bonds give them)") \
            from None
    n = cation.shape[0] if cation.ndim == 1 else -1
    if n < 0 or anion.shape != (n,) or vec_ang.shape != (n, 3) \
            or d_ang.shape != (n,):
        raise ValueError("bonds: cation, anion (n,), vec_ang (n, 3) and d_ang "
                         "(n,) need matching shapes")
    if n and (cation.dtype.kind not in "iu" or anion.dtype.kind not in "iu"):
        raise ValueError("bonds: cation and anion are atom rows (integers)")
    cation = cation.astype(np.int64)
    anion = anion.astype(np.int64)
    if n and (min(cation.min(), anion.min()) < 0
              or max(cation.max(), anion.max()) >= n_atoms):
        raise ValueError(f"bonds name atom rows outside 0 .. {n_atoms - 1}; "
                         "they belong to another frame")
    return cation, anion, vec_ang, d_ang


# ---------------------------------------------------------------------------
# the partial pair distribution functions
# ---------------------------------------------------------------------------

def rdf_grid_ang(frame: Frame, r_max_ang: float, dr_ang: float = RDF_DR_ANG
                 ) -> tuple[np.ndarray, tuple[str, ...]]:
    """The r grid of :func:`partial_rdf` for this frame, and its notes.

    ``arange(dr, r_max + dr/2, dr)`` as pdf.pair_distribution builds it
    (``pdf.py:440``), less any point beyond half the smallest perpendicular
    width of the box, with a note when the request was cut there. The pairs
    feeding it have to be searched to ``grid[-1] + dr``.
    """
    if not isinstance(frame, Frame):
        raise ValueError(f"a Frame is needed, not {type(frame).__name__}")
    top_ang = _positive(r_max_ang, "r_max_ang")
    step_ang = _positive(dr_ang, "dr_ang")
    half_ang = float(frame.perpendicular_widths_ang.min()) / 2.0
    grid_ang = np.arange(step_ang, top_ang + 0.5 * step_ang, step_ang)
    notes = []
    if grid_ang.size and grid_ang[-1] > half_ang:
        notes.append(
            f"the g(r) grid stops at {half_ang:.6g} Å, half the smallest "
            f"perpendicular width of the box, below the {top_ang:.6g} Å asked "
            "for: beyond it a shell around an atom overlaps its own periodic "
            "images")
        grid_ang = grid_ang[grid_ang <= half_ang]
    if grid_ang.size < 2:
        raise ValueError(
            f"the g(r) grid holds {grid_ang.size} point(s): r_max "
            f"{top_ang:.6g} Å, dr {step_ang:.6g} Å, half the smallest box "
            f"width {half_ang:.6g} Å; at least two points are needed")
    return grid_ang, tuple(notes)


@dataclass(frozen=True, eq=False)
class PartialRDF:
    """One frame's g_ab(r) and running coordination N_ab(r).

    ``g`` is the same array for (a, b) and (b, a). ``n_cum[k]`` is the mean
    number of b atoms within ``r_ang[k]`` of an a atom, counted exactly
    (d <= r_k). ``radial_per_ang`` is R_ab(r) = 4 pi r^2 rho_b g_ab(r), whose
    integral over a peak is the CN of that shell.
    """

    pair: tuple[str, str]
    r_ang: np.ndarray
    g: np.ndarray
    n_cum: np.ndarray
    n_centre: int
    n_neighbour: int
    volume_ang3: float
    dr_ang: float
    notes: tuple[str, ...] = ()

    @property
    def rho_neighbour_per_ang3(self) -> float:
        return self.n_neighbour / self.volume_ang3

    @property
    def radial_per_ang(self) -> np.ndarray:
        return 4.0 * math.pi * self.r_ang ** 2 * self.rho_neighbour_per_ang3 \
            * self.g

    @property
    def uncorrelated_level(self) -> float:
        """The large-r limit of g in this normalisation: 1, or 1 - 1/N_a for a
        pair of one element (each atom has N_a - 1 partners of its kind)."""
        if self.pair[0] == self.pair[1]:
            return (self.n_centre - 1) / self.n_centre
        return 1.0

    @property
    def pair_counts_per_unit_g(self) -> np.ndarray:
        """Per grid point, the independent pairs g = 1 stands for in this
        frame (:func:`pair_counts_per_unit_g`)."""
        return pair_counts_per_unit_g(
            self.r_ang, self.dr_ang, self.n_centre, self.n_neighbour,
            self.volume_ang3, same_element=self.pair[0] == self.pair[1],
            n_frames=1)


def pair_counts_per_unit_g(r_ang, dr_ang: float, n_centre: int,
                           n_neighbour: int, volume_ang3: float, *,
                           same_element: bool, n_frames: int) -> np.ndarray:
    """Per grid point, the number of independent pairs that g = 1 stands for.

    n_frames 4 pi r^2 dr N_a N_b / V, the deposit that the normalisation of
    :func:`partial_rdf` turns into g = 1, summed over the frames averaged;
    halved for a pair of one element, whose pairs are deposited from both
    ends. g times this is the number of pairs counted at that point, so the
    Poisson standard error of g is sqrt(g / this) (module docstring).
    """
    grid_ang = np.asarray(r_ang, dtype=np.float64)
    if grid_ang.ndim != 1 or not np.isfinite(grid_ang).all() \
            or (grid_ang <= 0).any():
        raise ValueError("r_ang needs a 1-D grid of distances above 0")
    step_ang = _positive(dr_ang, "dr_ang")
    volume = _positive(volume_ang3, "volume_ang3")
    counts = []
    for value, name in ((n_centre, "n_centre"), (n_neighbour, "n_neighbour"),
                        (n_frames, "n_frames")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} {value!r}: a whole number of 1 or more "
                             "is needed")
        counts.append(int(value))
    na, nb, frames = counts
    if same_element and na != nb:
        raise ValueError(f"a pair of one element has one atom count, not "
                         f"{na} and {nb}")
    out = frames * 4.0 * math.pi * grid_ang ** 2 * step_ang \
        * (float(na) * float(nb)) / volume
    return out / 2.0 if same_element else out


def _frozen(array) -> np.ndarray:
    out = np.array(array, copy=True)
    out.setflags(write=False)
    return out


def _first_at_or_above(grid_ang: np.ndarray, d_ang: np.ndarray) -> np.ndarray:
    """Per distance, the index of the first grid value >= it (len(grid) if none).

    ``numpy.searchsorted(grid, d, side='left')`` on a uniform grid, from
    arithmetic: the rounded index is within one step of the answer (the
    division errs by about 1e-13 of a step at 1 000 points), and one
    comparison each way against the grid values themselves makes it exact.
    On 1.7 million distances and a 1 000-point grid it took 0.08 s against
    0.15 s for the binary search (medians of 7, measured); a test checks it
    equals searchsorted, ties included.
    """
    n = grid_ang.shape[0]
    k = np.clip(np.ceil((d_ang - grid_ang[0]) / (grid_ang[1] - grid_ang[0])),
                0, n).astype(np.int64)
    inside = np.flatnonzero(k < n)
    k[inside[grid_ang[k[inside]] < d_ang[inside]]] += 1
    after = np.flatnonzero(k > 0)
    k[after[grid_ang[k[after] - 1] >= d_ang[after]]] -= 1
    return k


class _RdfAccumulator:
    """Deposits the pairs of one frame, block by block, for every element pair."""

    def __init__(self, frame: Frame, grid_ang: np.ndarray, step_ang: float):
        self.frame = frame
        species, inverse = np.unique(frame.elements, return_inverse=True)
        self.species = [str(s) for s in species]
        self.code = inverse.reshape(-1)
        self.n_el = len(self.species)
        self.count = np.bincount(self.code, minlength=self.n_el)
        self.grid_ang = np.asarray(grid_ang, dtype=np.float64)
        self.step_ang = float(step_ang)
        self.need_ang = float(self.grid_ang[-1] + self.step_ang)
        n_points = self.grid_ang.shape[0]
        self.deposit = np.zeros((self.n_el, self.n_el, n_points))
        self.first = np.zeros((self.n_el, self.n_el, n_points + 1),
                              dtype=np.int64)
        self.coverage = bulk._Coverage(frame, frame.n_atoms)

    def add(self, block: bulk.PairTable) -> None:
        self.coverage.add(block)
        if block.r_ang < self.need_ang:
            raise ValueError(
                f"the pairs were searched to {block.r_ang:.6g} Å; the g(r) grid "
                f"needs them to {self.need_ang:.6g} Å (its last point plus dr)")
        ci = self.code[block.i]
        cj = self.code[block.j]
        d_ang = block.d_ang
        # each unordered pair of two elements once, from the a <= b end; both
        # orders within one element, as R(r) sums over every j != i
        keep = (ci <= cj) & (d_ang <= self.need_ang)
        if not keep.any():
            return
        ci, cj, d_ang = ci[keep], cj[keep], d_ang[keep]
        combined = ci * self.n_el + cj
        first = _first_at_or_above(self.grid_ang, d_ang)
        n_points = self.grid_ang.shape[0]
        for c in np.unique(combined):
            sel = combined == c
            a, b = divmod(int(c), self.n_el)
            chosen_ang = d_ang[sel]
            self.deposit[a, b] += _deposit(self.grid_ang, chosen_ang,
                                           np.ones(chosen_ang.shape[0]))
            self.first[a, b] += np.bincount(first[sel],
                                            minlength=n_points + 1)

    def result(self) -> dict[tuple[str, str], PartialRDF]:
        self.coverage.check()
        volume = self.frame.volume_ang3
        grid_ang = _frozen(self.grid_ang)
        shell_ang3 = 4.0 * math.pi * self.grid_ang ** 2 * self.step_ang
        n_points = self.grid_ang.shape[0]
        out: dict[tuple[str, str], PartialRDF] = {}
        for a in range(self.n_el):
            for b in range(a, self.n_el):
                na, nb = int(self.count[a]), int(self.count[b])
                g = _frozen(self.deposit[a, b] * volume
                            / (shell_ang3 * (float(na) * float(nb))))
                cum = np.cumsum(self.first[a, b])[:n_points].astype(np.float64)
                sa, sb = self.species[a], self.species[b]
                notes = ()
                if a == b and na == 1:
                    notes = (f"the model holds one {sa} atom, so g_{sa}{sa}(r) "
                             "has no pair and is 0",)
                out[(sa, sb)] = PartialRDF((sa, sb), grid_ang, g,
                                           _frozen(cum / na), na, nb, volume,
                                           self.step_ang, notes)
                if a != b:
                    out[(sb, sa)] = PartialRDF((sb, sa), grid_ang, g,
                                               _frozen(cum / nb), nb, na,
                                               volume, self.step_ang, notes)
        return out


def partial_rdf(frame: Frame, pairs, *, r_max_ang: float,
                dr_ang: float = RDF_DR_ANG) -> dict[tuple[str, str], PartialRDF]:
    """g_ab(r) and N_ab(r) of every ordered element pair, from one pair search.

    ``pairs`` is a ``bulk.PairTable`` or the blocks of ``bulk.iter_pairs`` on
    this frame, covering every atom once and searched to at least the last
    grid point plus ``dr_ang`` (:func:`rdf_grid_ang` gives the grid). The
    module docstring gives the normalisation. Notes of the grid cap are on
    every result.
    """
    grid, notes = rdf_grid_ang(frame, r_max_ang, dr_ang)
    acc = _RdfAccumulator(frame, grid, float(dr_ang))
    for block in bulk._blocks_of(pairs):
        acc.add(block)
    out = acc.result()
    if notes:
        out = {key: PartialRDF(p.pair, p.r_ang, p.g, p.n_cum, p.n_centre,
                               p.n_neighbour, p.volume_ang3, p.dr_ang,
                               p.notes + notes) for key, p in out.items()}
    return out


# ---------------------------------------------------------------------------
# the first minimum
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MinimumMethod:
    """How a first minimum is located; every field is required and stated.

    ``rule``: ``'valley'`` or ``'first local minimum'`` (module docstring).
    ``smooth_sigma_ang``: None for the g(r) as measured, or the width of a
    Gaussian it is smoothed with first. ``flat_rule``: ``'first'`` or
    ``'midpoint'`` of the floor. ``margin_std_errors``: how many standard
    errors of g a difference has to exceed to count (a crossing of the
    uncorrelated level, the drop that ends a maximum, the rise that ends a
    minimum), and within how many of the lowest point the floor extends; 0
    compares the values as they stand. All are method choices, Sam's to set
    after looking at a g(r) of a real model; none has a default.
    """

    rule: str
    smooth_sigma_ang: float | None
    flat_rule: str
    margin_std_errors: float

    def __post_init__(self) -> None:
        if self.rule not in MINIMUM_RULES:
            raise ValueError(f"rule {self.rule!r} is not one of {MINIMUM_RULES}")
        if self.flat_rule not in FLAT_RULES:
            raise ValueError(f"flat_rule {self.flat_rule!r} is not one of "
                             f"{FLAT_RULES}")
        if self.smooth_sigma_ang is not None:
            object.__setattr__(self, "smooth_sigma_ang", _positive(
                self.smooth_sigma_ang, "smooth_sigma_ang"))
        object.__setattr__(self, "margin_std_errors", _non_negative(
            self.margin_std_errors, "margin_std_errors"))

    def describe(self) -> str:
        source = ("g(r) as measured (no smoothing)"
                  if self.smooth_sigma_ang is None else
                  f"g(r) smoothed by a Gaussian of sigma "
                  f"{self.smooth_sigma_ang!r} Å (cut at "
                  f"{SMOOTH_HALF_WIDTH_SIGMAS:g} sigma, normalised at the grid "
                  "ends)")
        if self.rule == "valley":
            how = ("the lowest point between the first maximum, once g has "
                   "risen above its uncorrelated level, and where g, having "
                   "fallen below that level, next rises above it")
        else:
            how = ("the first local minimum after the first local maximum "
                   "above the uncorrelated level")
        margin = self.margin_std_errors
        if margin == 0.0:
            judged = "values compared as they stand"
        else:
            judged = (f"a crossing, a drop or a rise counts when it exceeds "
                      f"{margin:g} standard error(s) of g, and the floor holds "
                      f"every point within {margin:g} standard error(s) of the "
                      "lowest")
        flat = "first point" if self.flat_rule == "first" else "midpoint"
        return (f"{self.rule} rule on {source}: {how}; {judged}; the floor "
                f"gives its {flat}; the point is a grid point")


@dataclass(frozen=True)
class RdfMinimum:
    """A first minimum of g_ab(r), or why there is none (``r_ang`` None).

    ``source`` is ``'auto'`` for a point the rule located, ``'user'`` for a
    cutoff the user gave (``analyse_trajectory`` then records the rule's own
    point in ``notes``). ``level`` is the uncorrelated level g was compared
    with; ``floor_r_ang`` / ``floor_index`` the first and last grid point of
    the floor; ``g_value`` and ``g_std_error`` g as measured at the point and
    its Poisson counting error, unsmoothed whatever the method (the error is
    None when no pair counts were given); ``depth_std_errors`` how far the
    lowest point of the g the rule reads (smoothed, when the method smooths)
    lies below the level, in standard errors of that g at the level, negative
    when it lies above.
    """

    pair: tuple[str, str]
    r_ang: float | None
    index: int | None
    g_value: float | None
    method: str
    source: str                        # 'auto' | 'user'
    reason: str = ""
    level: float | None = None
    floor_r_ang: tuple[float, float] | None = None
    floor_index: tuple[int, int] | None = None
    g_std_error: float | None = None
    depth_std_errors: float | None = None
    notes: tuple[str, ...] = ()

    @property
    def found(self) -> bool:
        return self.r_ang is not None


def _non_negative(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} {value!r} is not a number")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} {value!r} is not a number") from None
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} is {value!r}; a finite number of 0 or more "
                         "is needed")
    return out


def _kernel_half(step_ang: float, sigma_ang: float) -> int:
    """Grid points on each side of the centre of the smoothing kernel."""
    return max(int(math.ceil(SMOOTH_HALF_WIDTH_SIGMAS * sigma_ang / step_ang)),
               1)


def _kernel(n_points: int, step_ang: float, sigma_ang: float) -> np.ndarray:
    half = _kernel_half(step_ang, sigma_ang)
    if 2 * half + 1 > n_points:
        raise ValueError(
            f"a smoothing sigma of {sigma_ang!r} Å spans {2 * half + 1} grid "
            f"points, more than the {n_points} of the grid")
    offset_ang = np.arange(-half, half + 1) * step_ang
    return np.exp(-0.5 * (offset_ang / sigma_ang) ** 2)


def _smooth(values: np.ndarray, step_ang: float, sigma_ang: float
            ) -> np.ndarray:
    """Gaussian smoothing, normalised by the kernel weight inside the grid."""
    kernel = _kernel(values.shape[0], step_ang, sigma_ang)
    weight = np.convolve(np.ones_like(values), kernel, mode="same")
    return np.convolve(values, kernel, mode="same") / weight


def _smooth_variance(variance: np.ndarray, step_ang: float, sigma_ang: float
                     ) -> np.ndarray:
    """The variance of :func:`_smooth`'s output for independent points:
    sum w^2 var / (sum w)^2 over the same kernel weights."""
    kernel = _kernel(variance.shape[0], step_ang, sigma_ang)
    weight = np.convolve(np.ones_like(variance), kernel, mode="same")
    return np.convolve(variance, kernel ** 2, mode="same") / weight ** 2


def _held(side: np.ndarray, values: np.ndarray, level: float, per_unit,
          margin: float, sign: float) -> np.ndarray:
    """The points of the stretches of ``side`` (g above, or below, its level)
    whose pairs differ from the number expected at the level by more than
    ``margin`` Poisson standard errors of the larger of the two numbers.

    The pairs of a stretch are counted on the g as measured (g x per_unit
    at each point, summed), whatever the stretch was drawn on. A pair's hat
    weights inside a stretch sum to at most 1, so sqrt of the summed count is
    an upper bound of its standard error. Without pair counts every stretch
    is held (g taken as exact).
    """
    out = np.zeros(side.shape, dtype=bool)
    padded = np.concatenate(([False], side, [False]))
    change = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = change[0::2], change[1::2]
    if not starts.size:
        return out
    if per_unit is None:
        keep = np.ones(starts.shape, dtype=bool)
    else:
        seen = np.concatenate(([0.0], np.cumsum(values * per_unit)))
        expected = np.concatenate(([0.0], np.cumsum(level * per_unit)))
        sum_seen = seen[ends] - seen[starts]
        sum_expected = expected[ends] - expected[starts]
        keep = sign * (sum_seen - sum_expected) > margin * np.sqrt(
            np.maximum(np.maximum(sum_seen, sum_expected), 0.0))
    for start, end in zip(starts[keep], ends[keep]):
        out[start:end] = True
    return out


def first_minimum(r_ang, g, method: MinimumMethod, *,
                  pair: tuple[str, str] = ("", ""), level: float = 1.0,
                  pairs_per_unit_g=None) -> RdfMinimum:
    """The first minimum of one g(r) on a uniform grid, by ``method``.

    ``level``: the uncorrelated level of this g, 1 for a pair of two elements
    and 1 - 1/N_a for a pair of one element in this module's normalisation
    (``PartialRDF.uncorrelated_level``). ``pairs_per_unit_g``: per grid point,
    the pairs g = 1 stands for (:func:`pair_counts_per_unit_g`), from which
    the standard errors come; None takes g as exact (every error 0), which the
    result states. Returns an :class:`RdfMinimum` whose ``r_ang`` is a grid
    point, or None with the reason when the rule finds no such point.
    ValueError for a grid that is not uniform, increasing and as long as
    ``g``, a ``g`` that is not finite, or pair counts that are not positive.
    """
    if not isinstance(method, MinimumMethod):
        raise ValueError("method needs a MinimumMethod (no default rule)")
    grid_ang = np.asarray(r_ang, dtype=np.float64)
    values = np.asarray(g, dtype=np.float64)
    if grid_ang.ndim != 1 or values.shape != grid_ang.shape \
            or grid_ang.size < 3:
        raise ValueError("r_ang and g need the same 1-D shape, at least three "
                         "points")
    if not np.isfinite(values).all() or not np.isfinite(grid_ang).all():
        raise ValueError(f"g_{''.join(pair)}(r) or its grid holds NaN or "
                         "infinite values")
    steps_ang = np.diff(grid_ang)
    step_ang = float(steps_ang[0])
    # a uniform grid: np.arange steps differ by rounding only (about 1e-16
    # relative); 1e-6 relative is a parsing tolerance, not a physical value
    if step_ang <= 0 or not np.allclose(steps_ang, step_ang, rtol=1e-6,
                                        atol=0.0):
        raise ValueError("the r grid is not uniform and increasing")
    level = _non_negative(level, "level")
    if pairs_per_unit_g is None:
        per_unit = None
    else:
        per_unit = np.asarray(pairs_per_unit_g, dtype=np.float64)
        if per_unit.shape != grid_ang.shape or not np.isfinite(per_unit).all() \
                or (per_unit <= 0).any():
            raise ValueError("pairs_per_unit_g needs one finite value above 0 "
                             "per grid point")
    text = method.describe() + (
        "; standard errors from the pair counts at each grid point"
        if per_unit is not None else
        "; no pair counts given, so g is taken as exact (every standard "
        "error 0)")
    label = f"g_{pair[0]}{pair[1]}(r)" if pair[0] else "g(r)"
    margin = method.margin_std_errors
    by = "" if margin == 0.0 or per_unit is None else \
        f" by more than {margin:g} standard error(s)"
    top = f"{grid_ang[-1]:.6g} Å"

    def none(reason: str) -> RdfMinimum:
        return RdfMinimum(pair, None, None, None, text, "auto",
                          f"{label}: {reason}", level=level)

    if per_unit is None:
        var_seen = var_level = np.zeros_like(values)
    else:
        # Poisson: g times per_unit pairs counted, so var(g) = g / per_unit
        var_seen = np.maximum(values, 0.0) / per_unit
        var_level = level / per_unit
    smooth = values
    if method.smooth_sigma_ang is not None:
        sigma_ang = method.smooth_sigma_ang
        smooth = _smooth(values, step_ang, sigma_ang)
        var_seen = _smooth_variance(var_seen, step_ang, sigma_ang)
        var_level = _smooth_variance(var_level, step_ang, sigma_ang)
    err_seen = np.sqrt(var_seen)
    err_level = np.sqrt(var_level)
    # a rise above the level is judged point by point; a fall below it by
    # the whole stretch below the level (module docstring)
    above = smooth - level > margin * np.maximum(err_seen, err_level)
    below = _held(smooth < level, values, level, per_unit, margin, -1.0)
    n_points = values.shape[0]

    first_above = np.flatnonzero(above)
    if not first_above.size:
        return none(f"does not exceed {level:.6g}, its uncorrelated level,"
                    f"{by} within {top}")
    up = int(first_above[0])
    notes: list[str] = []
    if method.rule == "valley":
        falls = np.flatnonzero(below[up:])
        if not falls.size:
            return none(f"does not fall below {level:.6g}{by} within {top} "
                        f"after it first exceeds it at {grid_ang[up]:.6g} Å")
        down = up + int(falls[0])
        rises = np.flatnonzero(above[down:])
        if not rises.size:
            return none(f"falls below {level:.6g}{by} at "
                        f"{grid_ang[down]:.6g} Å and does not exceed it again"
                        f"{by} within {top}, so the valley has no upper end "
                        "on this grid")
        end = down + int(rises[0])
        # the valley starts after the last point above the level before the
        # fall, so a gap inside one shell, not below the level by the margin,
        # is not a valley
        start = up + int(np.flatnonzero(above[up:down])[-1]) + 1
        if end - down <= 2:
            notes.append(f"the valley of {label} below {level:.6g} spans "
                         f"{end - down} grid point(s), "
                         f"{grid_ang[down]:.6g}-{grid_ang[end - 1]:.6g} Å")
    else:
        peak, start = up, None
        for j in range(up + 1, n_points):
            if smooth[j] > smooth[peak]:
                peak = j
            elif smooth[peak] - smooth[j] > margin * math.hypot(
                    err_seen[peak], err_seen[j]):
                start = j
                break
        if start is None:
            return none(f"has no local maximum above {level:.6g} followed by "
                        f"a drop{by} within {top}")
        low, end = start, None
        for j in range(start + 1, n_points):
            if smooth[j] < smooth[low]:
                low = j
            elif smooth[j] - smooth[low] > margin * math.hypot(
                    err_seen[low], err_seen[j]):
                end = j
                break
        if end is None:
            return none(f"has no local minimum after its first maximum at "
                        f"{grid_ang[peak]:.6g} Å{by} within {top}")
        start = peak + 1
    window = np.arange(start, end)
    lowest = int(window[np.argmin(smooth[window])])
    within = margin * np.sqrt(var_seen[window] + var_seen[lowest])
    floor = window[smooth[window] - smooth[lowest] <= within]
    lo, hi = int(floor[0]), int(floor[-1])
    index = lo if method.flat_rule == "first" else (lo + hi) // 2
    if smooth[lowest] > level:
        notes.append(f"{label} at its first local minimum, "
                     f"{grid_ang[lowest]:.6g} Å, is {smooth[lowest]:.4g}, above "
                     f"its uncorrelated level {level:.6g}")
    depth = None
    if per_unit is not None and err_level[lowest] > 0:
        depth = float((level - smooth[lowest]) / err_level[lowest])
    # g_value is g as measured, so its error is the unsmoothed one too
    g_error = None if per_unit is None else math.sqrt(
        max(float(values[index]), 0.0) / float(per_unit[index]))
    return RdfMinimum(
        pair, float(grid_ang[index]), index, float(values[index]), text,
        "auto", level=level,
        floor_r_ang=(float(grid_ang[lo]), float(grid_ang[hi])),
        floor_index=(lo, hi), g_std_error=g_error,
        depth_std_errors=depth, notes=tuple(notes))


# ---------------------------------------------------------------------------
# bonds cut by distance
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class DistanceBonds:
    """Cation-anion pairs with d <= the cutoff of their element pair, each once.

    The same fields as ``bulk.Bonds`` (from the cation's end: ``vec_ang``
    points from the cation to the anion's image), so every descriptor takes
    either. ``cutoffs_ang`` is keyed (cation element, anion element);
    ``missing`` lists the cation-anion element pairs of the model that have no
    cutoff, whose contacts are therefore in no bond.
    """

    cation: np.ndarray
    anion: np.ndarray
    image: np.ndarray
    vec_ang: np.ndarray
    d_ang: np.ndarray
    cutoffs_ang: dict
    missing: tuple

    def __len__(self) -> int:
        return int(self.cation.shape[0])


def _roles(symbols: np.ndarray, ox: np.ndarray) -> tuple[tuple[str, ...],
                                                       tuple[str, ...]]:
    """(cation elements, anion elements), alphabetical, from the state signs."""
    cations = {str(s) for s in np.unique(symbols[ox >= 0])}
    anions = {str(s) for s in np.unique(symbols[ox < 0])}
    both = sorted(cations & anions)
    if both:
        raise ValueError(
            f"{', '.join(both)} carry both cationic and anionic states in this "
            "model; element-pair cutoffs and descriptors need each element on "
            "one side")
    return tuple(sorted(cations)), tuple(sorted(anions))


def _cation_anion_cutoffs(cutoffs_ang, cations, anions
                          ) -> tuple[dict, list[str]]:
    """{(cation, anion): cutoff}, either key order accepted; and notes."""
    out: dict[tuple[str, str], float] = {}
    notes = []
    for key, value in dict(cutoffs_ang).items():
        if not (isinstance(key, tuple) and len(key) == 2):
            raise ValueError(f"cutoff key {key!r}: an element pair is needed")
        x, y = validate_symbol(str(key[0])), validate_symbol(str(key[1]))
        cutoff = _positive(value, f"the {x}-{y} cutoff")
        if x in cations and y in anions:
            pair = (x, y)
        elif x in anions and y in cations:
            pair = (y, x)
        elif {x, y} <= set(cations) | set(anions):
            raise ValueError(
                f"a cutoff for {x}-{y} joins two "
                f"{'cations' if x in cations else 'anions'}; a distance bond "
                "here is a cation-anion pair")
        else:
            notes.append(f"the cutoff given for {x}-{y} is not used: the model "
                         f"holds no {x if x not in cations + anions else y}")
            continue
        if pair in out and out[pair] != cutoff:
            raise ValueError(f"two different cutoffs are given for {pair[0]}-"
                             f"{pair[1]} ({out[pair]!r} and {cutoff!r} Å)")
        out[pair] = cutoff
    return out, notes


class _DistanceAccumulator:
    """Collects the cation-anion pairs within their cutoffs, block by block."""

    def __init__(self, frame: Frame, ox: np.ndarray,
                 cutoffs: Mapping[tuple[str, str], float]):
        self.frame = frame
        symbols = frame.elements
        species, inverse = np.unique(symbols, return_inverse=True)
        position = {str(s): n for n, s in enumerate(species)}
        self.code = inverse.reshape(-1)
        self.is_cation = ox >= 0
        self.limit_ang = np.full((len(species), len(species)), np.nan)
        for (c, a), value in cutoffs.items():
            if c in position and a in position:
                self.limit_ang[position[c], position[a]] = value
        self.max_cut_ang = max(cutoffs.values()) if cutoffs else 0.0
        self.coverage = bulk._Coverage(frame, frame.n_atoms)
        self.parts: list[tuple] = []
        self.cutoffs = dict(cutoffs)

    def add(self, block: bulk.PairTable) -> None:
        self.coverage.add(block)
        if self.max_cut_ang > block.r_ang:
            raise ValueError(f"a distance cutoff of {self.max_cut_ang!r} Å "
                             f"lies beyond the {block.r_ang!r} Å the pairs "
                             "were searched to")
        i, j, d_ang = block.i, block.j, block.d_ang
        candidate = np.flatnonzero(self.is_cation[i] & ~self.is_cation[j])
        if not candidate.size:
            return
        limit_ang = self.limit_ang[self.code[i[candidate]],
                                   self.code[j[candidate]]]
        with np.errstate(invalid="ignore"):
            take = candidate[d_ang[candidate] <= limit_ang]  # NaN: no cutoff
        if not take.size:
            return
        if block.vec_ang is None:
            raise ValueError("the pairs carry no vectors; distance bonds need "
                             "them (vectors_within_ang)")
        vec_ang = block.vec_ang[take]
        if np.isnan(vec_ang).any():
            raise ValueError(f"the pairs carry vectors only within "
                             f"{block.vectors_within_ang!r} Å; the distance "
                             f"bonds need them to {self.max_cut_ang!r} Å")
        self.parts.append((i[take], j[take], block.image[take], vec_ang,
                           d_ang[take]))

    def result(self, cations, anions) -> DistanceBonds:
        self.coverage.check()
        if self.parts:
            i, j, image, vec_ang, d_ang = (np.concatenate(p)
                                           for p in zip(*self.parts))
        else:
            i = j = np.zeros(0, dtype=np.int32)
            image = np.zeros((0, 3), dtype=np.int16)
            vec_ang = np.zeros((0, 3))
            d_ang = np.zeros(0)
        order = np.lexsort((image[:, 2], image[:, 1], image[:, 0], j, i))
        present = set(self.frame.species)
        missing = tuple((c, a) for c in cations for a in anions
                        if c in present and a in present
                        and (c, a) not in self.cutoffs)
        return DistanceBonds(
            cation=_frozen(i[order].astype(np.int32)),
            anion=_frozen(j[order].astype(np.int32)),
            image=_frozen(image[order]), vec_ang=_frozen(vec_ang[order]),
            d_ang=_frozen(d_ang[order]), cutoffs_ang=dict(sorted(
                self.cutoffs.items())), missing=missing)


def distance_bonds(frame: Frame, pairs, ox_atom,
                   cutoffs_ang: Mapping[tuple[str, str], float]
                   ) -> DistanceBonds:
    """The cation-anion bonds by distance: d <= the cutoff of the element pair.

    ``cutoffs_ang`` maps an element pair, in either order, to a cutoff in Å
    (inclusive, as ``SiteResult.cn_within``); a pair of two cations or two
    anions is refused. ``pairs`` are the blocks of ``bulk.iter_pairs`` on this
    frame, searched at least to the largest cutoff with vectors that far. The
    cation-anion element pairs left without a cutoff are listed in
    ``missing``; their contacts are in no bond.
    """
    if not isinstance(frame, Frame):
        raise ValueError(f"a Frame is needed, not {type(frame).__name__}")
    ox = bulk._ox_array(ox_atom, frame.elements)
    cations, anions = _roles(frame.elements, ox)
    cutoffs, _ = _cation_anion_cutoffs(cutoffs_ang, cations, anions)
    acc = _DistanceAccumulator(frame, ox, cutoffs)
    for block in bulk._blocks_of(pairs):
        acc.add(block)
    return acc.result(cations, anions)


# ---------------------------------------------------------------------------
# coordination numbers
# ---------------------------------------------------------------------------

def bond_cn(bonds, n_atoms: int) -> np.ndarray:
    """Per atom, its number of bonds: cations count anions, anions cations.

    From ``bulk.bonds_at`` this equals ``AtomResults.cn`` atom for atom (the
    tests check it); from :class:`DistanceBonds` it is the distance-cut CN
    over the counter-ions that have a cutoff.
    """
    cation, anion, _, _ = _bond_arrays(bonds, n_atoms)
    return (np.bincount(cation, minlength=n_atoms)
            + np.bincount(anion, minlength=n_atoms)).astype(np.int64)


def cn_counts(cn, elements, element: str) -> dict[int, int]:
    """CN value -> number of atoms of ``element`` with it."""
    symbols = _symbols(elements)
    values = np.asarray(cn)
    if values.shape != symbols.shape:
        raise ValueError(f"cn has shape {values.shape} for {symbols.size} atoms")
    element = validate_symbol(element)
    found, counts = np.unique(values[symbols == element], return_counts=True)
    return {int(v): int(c) for v, c in zip(found, counts)}


def cn_groups(cn, elements, element: str, values: Sequence[int]
              ) -> dict[str, int]:
    """Atoms of ``element`` with each CN in ``values``, and every other CN.

    Keys ``f'{element}{v}'`` for each value and ``'other'``, which is
    reported, never folded into a named group.
    """
    element = validate_symbol(element)
    wanted = [int(v) for v in values]
    counts = cn_counts(cn, elements, element)
    out = {f"{element}{v}": counts.get(v, 0) for v in wanted}
    out["other"] = sum(c for v, c in counts.items() if v not in wanted)
    return out


def boron_n4(cn, elements) -> dict[str, int]:
    """'B3', 'B4' and 'other': N4 is the B4 fraction of all B."""
    return cn_groups(cn, elements, "B", (3, 4))


def aluminium_cn(cn, elements) -> dict[str, int]:
    """'Al4', 'Al5', 'Al6' and 'other'."""
    return cn_groups(cn, elements, "Al", (4, 5, 6))


# ---------------------------------------------------------------------------
# the network: speciation, Q^n, Q^n(mX), connectivity, linkages
# ---------------------------------------------------------------------------

def former_bond_counts(bonds, elements, formers: Collection[str]
                       ) -> np.ndarray:
    """Per atom, the number of its bonds to a former (anion rows; 0 elsewhere)."""
    symbols = _symbols(elements)
    former_set = _formers(formers)
    cation, anion, _, _ = _bond_arrays(bonds, symbols.shape[0])
    to_former = np.isin(symbols[cation], list(former_set))
    return np.bincount(anion[to_former],
                       minlength=symbols.shape[0]).astype(np.int64)


def anion_speciation(bonds, elements, formers: Collection[str],
                     anion: str = "O") -> dict[str, int]:
    """Atoms of ``anion`` by their number of bonded formers.

    ``'free'`` 0, ``'NBO'`` 1, ``'BO'`` exactly 2, ``'tricluster'`` 3 or more.
    """
    symbols = _symbols(elements)
    anion = validate_symbol(anion)
    count = former_bond_counts(bonds, symbols, formers)[symbols == anion]
    return {"free": int((count == 0).sum()), "NBO": int((count == 1).sum()),
            "BO": int((count == 2).sum()),
            "tricluster": int((count >= 3).sum())}


def _bridges(bonds, symbols: np.ndarray, former_set: frozenset[str],
             bridging_anions: frozenset[str]):
    """(cation, anion, bridge) arrays: bridge marks bonds from a former to a
    bridging anion (one bonded to two or more formers)."""
    cation, anion, _, _ = _bond_arrays(bonds, symbols.shape[0])
    to_former = np.isin(symbols[cation], list(former_set))
    per_anion = np.bincount(anion[to_former], minlength=symbols.shape[0])
    bridging = np.isin(symbols, list(bridging_anions)) & (per_anion >= 2)
    return cation, anion, to_former, to_former & bridging[anion]


def qn_counts(bonds, elements, formers: Collection[str],
              bridging_anions: Collection[str]) -> dict[str, dict[int, int]]:
    """Per former element present: n -> number of its atoms with n bridges.

    n counts the former's bonds to anions of ``bridging_anions`` that are
    bonded to two or more formers (BO and tricluster).
    """
    symbols = _symbols(elements)
    former_set = _formers(formers)
    anions = _symbol_set(bridging_anions, "bridging_anions")
    cation, _, _, bridge = _bridges(bonds, symbols, former_set, anions)
    n_bridge = np.bincount(cation[bridge], minlength=symbols.shape[0])
    out = {}
    for element in sorted(former_set):
        rows = symbols == element
        if rows.any():
            found, counts = np.unique(n_bridge[rows], return_counts=True)
            out[element] = {int(v): int(c) for v, c in zip(found, counts)}
    return out


def _qnm_label(n: int, m: int, other: str) -> str:
    return f"Q{n}({m}{other})"


def qn_mx_counts(bonds, elements, formers: Collection[str],
                 bridging_anions: Collection[str]
                 ) -> dict[tuple[str, str], dict[str, int]]:
    """Per (former T, former X): ``'Q{n}({m}{X})'`` -> number of T atoms.

    n is T's Q^n; m the number of its T-anion-X linkages over its bridging
    anions (the other X formers on each, T itself not counted).
    """
    symbols = _symbols(elements)
    former_set = _formers(formers)
    anions = _symbol_set(bridging_anions, "bridging_anions")
    n_atoms = symbols.shape[0]
    cation, anion, to_former, bridge = _bridges(bonds, symbols, former_set,
                                                anions)
    n_bridge = np.bincount(cation[bridge], minlength=n_atoms)
    present = [e for e in sorted(former_set) if (symbols == e).any()]
    out = {}
    for other in present:
        on_anion = np.bincount(anion[to_former & (symbols[cation] == other)],
                               minlength=n_atoms)
        share = on_anion[anion[bridge]] - (symbols[cation[bridge]] == other)
        m = np.bincount(cation[bridge], weights=share.astype(np.float64),
                        minlength=n_atoms).astype(np.int64)
        for element in present:
            rows = np.flatnonzero(symbols == element)
            labels, counts = np.unique(
                np.stack([n_bridge[rows], m[rows]], axis=1), axis=0,
                return_counts=True)
            out[(element, other)] = {_qnm_label(int(n), int(k), other):
                                     int(c) for (n, k), c in
                                     zip(labels, counts)}
    return out


def connectivity(bonds, elements, formers: Collection[str],
                 bridging_anions: Collection[str]) -> dict[str, float]:
    """Mean n of Q^n per former element present, and over every former atom
    (key ``'all formers'``)."""
    qn = qn_counts(bonds, elements, formers, bridging_anions)
    out = {}
    total_n, total_atoms = [], 0
    for element, counts in qn.items():
        atoms = sum(counts.values())
        out[element] = math.fsum(n * c for n, c in counts.items()) / atoms
        total_n.extend(n * c for n, c in counts.items())
        total_atoms += atoms
    if total_atoms:
        out["all formers"] = math.fsum(total_n) / total_atoms
    return out


def linkage_counts(bonds, elements, anion: str = "O"
                   ) -> dict[tuple[str, str, str], int]:
    """(X, anion, Y) -> number of X-anion-Y linkages, X <= Y alphabetically.

    Each pair of cations bonded to one anion atom is one linkage; an anion
    bonded to k cations gives k(k - 1)/2. Every cation element bonded to an
    anion of this element is a key, with 0 where a pair does not occur.
    """
    symbols = _symbols(elements)
    anion = validate_symbol(anion)
    cation, anion_row, _, _ = _bond_arrays(bonds, symbols.shape[0])
    here = symbols[anion_row] == anion
    cation, anion_row = cation[here], anion_row[here]
    kinds = sorted({str(s) for s in symbols[cation]})
    if not kinds:
        return {}
    code = np.searchsorted(np.array(kinds), symbols[cation])
    table = np.zeros((symbols.shape[0], len(kinds)), dtype=np.int64)
    np.add.at(table, (anion_row, code), 1)
    out = {}
    for x in range(len(kinds)):
        for y in range(x, len(kinds)):
            if x == y:
                value = int((table[:, x] * (table[:, x] - 1) // 2).sum())
            else:
                value = int((table[:, x] * table[:, y]).sum())
            out[(kinds[x], anion, kinds[y])] = value
    return out


def anion_environments(bonds, elements, anion: str) -> dict[str, int]:
    """Environment label -> number of atoms of ``anion`` with it.

    ``'F-Al1Na2'``: the bonded cations by element, alphabetical; an anion
    bonded to no cation is ``'F-none'``.
    """
    symbols = _symbols(elements)
    anion = validate_symbol(anion)
    cation, anion_row, _, _ = _bond_arrays(bonds, symbols.shape[0])
    rows = np.flatnonzero(symbols == anion)
    if not rows.size:
        return {}
    here = symbols[anion_row] == anion
    kinds = sorted({str(s) for s in symbols[cation[here]]})
    table = np.zeros((symbols.shape[0], max(1, len(kinds))), dtype=np.int64)
    if kinds:
        code = np.searchsorted(np.array(kinds), symbols[cation[here]])
        np.add.at(table, (anion_row[here], code), 1)
    unique, counts = np.unique(table[rows], axis=0, return_counts=True)
    out = {}
    for row, count in zip(unique, counts):
        parts = "".join(f"{kinds[c]}{int(v)}" for c, v in enumerate(row)
                        if kinds and v)
        out[f"{anion}-{parts or 'none'}"] = int(count)
    return out


def lone_pair_values(results: bulk.AtomResults, elements, element: str
                     ) -> dict[str, np.ndarray]:
    """phi, CN and plateau width (decades) over every atom of ``element``."""
    symbols = _symbols(elements)
    element = validate_symbol(element)
    rows = symbols == element
    return {"phi": np.asarray(results.phi)[rows].copy(),
            "cn": np.asarray(results.cn)[rows].copy(),
            "plateau_decades": np.asarray(results.plateau_decades)[rows].copy()}


# ---------------------------------------------------------------------------
# angles and lengths
# ---------------------------------------------------------------------------

def bond_angles_deg(bonds, elements, centre: str,
                    ends: Collection[str] | None = None
                    ) -> dict[tuple[str, str, str], np.ndarray]:
    """Every A-centre-C angle between two bonds of one centre atom, in degrees.

    The arms are the bonds of each atom of ``centre`` (its anions for a
    cation, its cations for an anion), restricted to other ends of element in
    ``ends`` when given. Keys (A, centre, C) with A <= C alphabetically.
    T-O-T is ``bond_angles_deg(b, el, 'O', ends=formers)``, O-T-O is
    ``bond_angles_deg(b, el, T, ends={'O'})``. atan2(|u x v|, u . v).
    """
    symbols = _symbols(elements)
    centre = validate_symbol(centre)
    end_set = None if ends is None else _symbol_set(ends, "ends")
    cation, anion, vec_ang, _ = _bond_arrays(bonds, symbols.shape[0])
    from_cation = symbols[cation] == centre
    from_anion = symbols[anion] == centre
    middle = np.concatenate([cation[from_cation], anion[from_anion]])
    other = np.concatenate([anion[from_cation], cation[from_anion]])
    arms = np.concatenate([vec_ang[from_cation], -vec_ang[from_anion]])
    if end_set is not None:
        keep = np.isin(symbols[other], list(end_set))
        middle, other, arms = middle[keep], other[keep], arms[keep]
    if middle.shape[0] < 2:
        return {}
    order = np.lexsort((other, middle))
    middle, other, arms = middle[order], other[order], arms[order]
    centres, start, degree = np.unique(middle, return_index=True,
                                       return_counts=True)
    width = int(degree.max())
    slot = np.arange(middle.shape[0]) - np.repeat(start, degree)
    row = np.repeat(np.arange(centres.shape[0]), degree)
    padded = np.zeros((centres.shape[0], width, 3))
    ends_pad = np.full((centres.shape[0], width), -1, dtype=np.int64)
    padded[row, slot] = arms
    ends_pad[row, slot] = other
    species, inverse = np.unique(symbols, return_inverse=True)
    species = [str(s) for s in species]
    code_pad = inverse.reshape(-1)[ends_pad]          # pads read row -1: unused
    collected: dict[tuple[str, str, str], list[np.ndarray]] = {}
    for a in range(width):
        for b in range(a + 1, width):
            ok = degree > b
            if not ok.any():
                continue
            u, v = padded[ok, a], padded[ok, b]
            angle = np.degrees(np.arctan2(np.linalg.norm(np.cross(u, v), axis=1),
                                          (u * v).sum(axis=1)))
            low = np.minimum(code_pad[ok, a], code_pad[ok, b])
            high = np.maximum(code_pad[ok, a], code_pad[ok, b])
            key = low * len(species) + high
            for value in np.unique(key):
                x, y = divmod(int(value), len(species))
                collected.setdefault((species[x], centre, species[y]),
                                     []).append(angle[key == value])
    return {key: np.concatenate(parts) for key, parts in sorted(
        collected.items())}


def bond_lengths_ang(bonds, elements) -> dict[tuple[str, str], np.ndarray]:
    """(cation element, anion element) -> the lengths of those bonds, Å."""
    symbols = _symbols(elements)
    cation, anion, _, d_ang = _bond_arrays(bonds, symbols.shape[0])
    out = {}
    if not cation.size:
        return out
    species, inverse = np.unique(symbols, return_inverse=True)
    inverse = inverse.reshape(-1)
    key = inverse[cation] * len(species) + inverse[anion]
    for value in np.unique(key):
        c, a = divmod(int(value), len(species))
        out[(str(species[c]), str(species[a]))] = \
            d_ang[key == value].copy()
    return out


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelComposition:
    """What the model holds, and the charge its oxidation states give."""

    counts: dict[str, int]
    n_atoms: int
    atomic_percent: dict[str, float]
    ox: dict[str, int]
    net_charge_e: int                  # exact: integer states on whole atoms
    neutral: bool
    mass_amu: float
    model_charge_e: float | None       # sum of the file's per-atom charges
    oxide_basis: dict[str, str] | None
    oxide_mol_percent: dict[str, float] | None
    anion_unassigned: dict[str, float] | None
    notes: tuple[str, ...] = ()


def _atomic_mass_constant_g() -> float:
    from scipy import constants

    return constants.physical_constants["atomic mass constant"][0] \
        / constants.gram


def _weight_amu(symbol: str) -> float:
    import gemmi

    return float(gemmi.Element(symbol).weight)


def density_g_per_cm3(frame: Frame) -> float:
    """Mass over volume: gemmi's standard atomic weights, scipy's atomic mass
    constant, and 1 Å = 1e-8 cm (scipy.constants.angstrom / centi)."""
    from scipy import constants

    if not isinstance(frame, Frame):
        raise ValueError(f"a Frame is needed, not {type(frame).__name__}")
    mass_amu = math.fsum(_weight_amu(el) * n
                         for el, n in frame.composition.items())
    volume_cm3 = frame.volume_ang3 * (constants.angstrom / constants.centi) ** 3
    return mass_amu * _atomic_mass_constant_g() / volume_cm3


def composition(frame: Frame, ox: ModelOxidation,
                oxide_basis: Mapping[str, str] | None = None
                ) -> ModelComposition:
    """Counts, atomic %, exact net charge, mass, and oxide mol % for a basis.

    ``oxide_basis`` maps each cation element of the model to the formula of
    its oxide (``{'Si': 'SiO2', 'Na': 'Na2O'}``); a formula may hold any of the
    model's anions (``'AlF3'``). Refused: a cation without a formula, a
    formula without its cation, or with another cation, or an element the
    model does not hold. The anions the oxides do not account for are
    ``anion_unassigned`` (negative when the basis needs more than the model
    holds), computed in exact rational arithmetic.
    """
    from .quality import parse_formula

    if not isinstance(frame, Frame):
        raise ValueError(f"a Frame is needed, not {type(frame).__name__}")
    if not isinstance(ox, ModelOxidation):
        raise ValueError("ox needs a ModelOxidation (md_model.model_oxidation)")
    counts = frame.composition
    n = frame.n_atoms
    ox.per_atom(frame.elements)                     # every element has a state
    states = {el: int(ox.ox[el]) for el in counts}
    net = sum(states[el] * c for el, c in counts.items())
    check_fractions([c / n for c in counts.values()], "atomic fractions")
    atomic = {el: 100.0 * c / n for el, c in counts.items()}
    notes = []
    weights = {el: _weight_amu(el) for el in counts}
    no_weight = sorted(el for el, w in weights.items() if not w > 0)
    if no_weight:
        notes.append(f"gemmi gives no standard atomic weight for "
                     f"{', '.join(no_weight)}; the mass and density leave "
                     "them out")
    mass = math.fsum(weights[el] * c for el, c in counts.items()
                     if el not in no_weight)
    if net:
        notes.append(f"the model is not neutral with these oxidation states: "
                     f"net charge {net:+d} e over {n} atoms ({ox.describe()})")
    model_charge = None
    if frame.charge_e is not None:
        model_charge = math.fsum(frame.charge_e.tolist())
        notes.append(f"the file's per-atom charges sum to {model_charge!r} e")

    basis_out = mol_percent = unassigned = None
    if oxide_basis is not None:
        basis_out, mol_percent, unassigned, more = _oxides(
            oxide_basis, counts, states, parse_formula)
        notes.extend(more)
    return ModelComposition(
        counts=dict(counts), n_atoms=n, atomic_percent=atomic, ox=states,
        net_charge_e=int(net), neutral=net == 0, mass_amu=mass,
        model_charge_e=model_charge, oxide_basis=basis_out,
        oxide_mol_percent=mol_percent, anion_unassigned=unassigned,
        notes=tuple(notes))


def _oxides(oxide_basis, counts, states, parse_formula):
    if isinstance(oxide_basis, (str, bytes)) or not isinstance(oxide_basis,
                                                               Mapping):
        raise ValueError("oxide_basis maps each cation element to the formula "
                         "of its oxide, e.g. {'Si': 'SiO2', 'Na': 'Na2O'}")
    basis: dict[str, tuple[str, dict[str, Fraction]]] = {}
    for key, formula in oxide_basis.items():
        element = validate_symbol(str(key))
        if element not in counts:
            raise ValueError(f"the oxide basis names {element}, which the model "
                             "does not hold")
        if states[element] < 0:
            raise ValueError(f"the oxide basis names {element}, an anion of "
                             "this model; it maps cations to their oxides")
        parsed = parse_formula(formula) if isinstance(formula, str) else {}
        if not parsed:
            raise ValueError(f"the formula given for {element} ({formula!r}) "
                             "could not be read")
        stoich = {validate_symbol(s): Fraction(repr(float(v)))
                  for s, v in parsed.items()}
        if element not in stoich:
            raise ValueError(f"the formula given for {element} ({formula}) "
                             f"holds no {element}")
        for other in stoich:
            if other == element:
                continue
            if other not in counts or states[other] >= 0:
                raise ValueError(f"{formula}: {other} is not an anion of this "
                                 "model; an oxide formula holds its cation "
                                 "and the model's anions")
        basis[element] = (str(formula), stoich)
    missing = sorted(el for el in counts if states[el] >= 0 and el not in basis)
    if missing:
        raise ValueError(f"the oxide basis has no formula for "
                         f"{', '.join(missing)}; every cation element of the "
                         "model needs one")
    moles = {el: Fraction(counts[el]) / stoich[el]
             for el, (_, stoich) in basis.items()}
    total = sum(moles.values())
    fractions = {basis[el][0]: moles[el] / total for el in moles}
    check_fractions([float(f) for f in fractions.values()],
                    "oxide mole fractions")
    mol_percent = {name: float(100 * f) for name, f in fractions.items()}
    anions = sorted(el for el in counts if states[el] < 0)
    unassigned = {}
    notes = []
    for a in anions:
        implied = sum(moles[el] * stoich.get(a, Fraction(0))
                      for el, (_, stoich) in basis.items())
        left = Fraction(counts[a]) - implied
        unassigned[a] = float(left)
        if left:
            notes.append(f"the oxide basis accounts for {float(implied)!r} of "
                         f"the model's {counts[a]} {a} atoms; "
                         f"{float(left)!r} are left "
                         f"{'unassigned' if left > 0 else 'short'}")
    return ({el: name for el, (name, _) in basis.items()}, mol_percent,
            unassigned, notes)


# ---------------------------------------------------------------------------
# the trajectory
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HistogramBins:
    """The bin widths of the continuous descriptors. All required: a bin width
    is a method choice with no physical default, stated in the result."""

    angle_deg: float
    phi: float
    plateau_decades: float
    length_ang: float

    def __post_init__(self) -> None:
        for name in ("angle_deg", "phi", "plateau_decades", "length_ang"):
            object.__setattr__(self, name, _positive(getattr(self, name),
                                                     f"bin width {name}"))


def _fixed_edges(width: float, top: float) -> np.ndarray:
    """0, width, 2 width, ... up to the first multiple at or above ``top``.

    The count is ceil(top / width), one more when the rounded product falls
    short of ``top``, so a value equal to ``top`` is always inside the last,
    closed bin; at most one extra empty bin results.
    """
    count = max(1, int(math.ceil(top / width)))
    if width * count < top:
        count += 1
    return width * np.arange(count + 1, dtype=np.float64)


class _Collector:
    """Per-frame values of every descriptor, built into md_stats containers."""

    def __init__(self) -> None:
        self.specs: dict[str, dict] = {}

    def _spec(self, name: str, kind: str, **meta) -> dict:
        spec = self.specs.get(name)
        if spec is None:
            spec = self.specs[name] = {"kind": kind, "rows": {}, **meta}
        return spec

    def counts(self, name, frame, counts, *, kind, keys=None, order=None):
        self._spec(name, "distribution", dkind=kind, keys=keys,
                   order=order)["rows"][frame] = dict(counts)

    def binned(self, name, frame, samples, edges, unit):
        spec = self._spec(name, "binned", edges=edges, unit=unit)
        spec["rows"][frame] = _bin(np.asarray(samples, dtype=np.float64),
                                   spec["edges"])

    def samples(self, name, frame, samples, unit, width):
        self._spec(name, "samples", unit=unit, width=width)["rows"][frame] = \
            np.asarray(samples, dtype=np.float64)

    def scalar(self, name, frame, value, unit):
        self._spec(name, "scalar", unit=unit)["rows"][frame] = float(value)

    def series(self, name, frame, values, axis, axis_name, axis_unit,
               value_unit):
        self._spec(name, "series", axis=axis, axis_name=axis_name,
                   axis_unit=axis_unit, value_unit=value_unit)["rows"][frame] \
            = np.asarray(values, dtype=np.float64)

    def build(self, empty: list[str]) -> dict[str, object]:
        out: dict[str, object] = {}
        for name, spec in self.specs.items():
            labels = sorted(spec["rows"])
            rows = [spec["rows"][k] for k in labels]
            kind = spec["kind"]
            if kind == "distribution":
                seen = set().union(*(r.keys() for r in rows))
                keys = spec["keys"]
                if keys == "range":
                    keys = tuple(range(0, max(seen) + 1)) if seen else None
                elif spec["order"] is not None:
                    keys = tuple(sorted(seen, key=spec["order"]))
                if not seen and not keys:
                    empty.append(name)
                    continue
                out[name] = Distribution.from_counts(
                    rows, name=name, kind=spec["dkind"], keys=keys,
                    frames=labels)
            elif kind == "binned":
                if not any(row[0].sum() + row[1] + row[2] + row[3] for row in rows):
                    empty.append(name)
                    continue
                out[name] = Histogram(
                    name, spec["unit"], spec["edges"],
                    np.stack([row[0] for row in rows]),
                    n_below=np.array([row[1] for row in rows], dtype=np.int64),
                    n_above=np.array([row[2] for row in rows], dtype=np.int64),
                    n_nonfinite=np.array([row[3] for row in rows],
                                         dtype=np.int64),
                    frames=labels)
            elif kind == "samples":
                if not any(row.size for row in rows):
                    empty.append(name)
                    continue
                finite = [row[np.isfinite(row)] for row in rows]
                top = max((float(f.max()) for f in finite if f.size),
                          default=0.0)
                edges = _fixed_edges(spec["width"], max(top, spec["width"]))
                out[name] = Histogram.from_samples(
                    rows, edges, name=name, unit=spec["unit"], frames=labels)
            elif kind == "scalar":
                out[name] = Scalar(name, spec["unit"], rows, frames=labels)
            else:
                out[name] = Series.from_frames(
                    spec["axis"], rows, name=name, axis_name=spec["axis_name"],
                    axis_unit=spec["axis_unit"],
                    value_unit=spec["value_unit"], frames=labels)
        return out


def _qnm_order(label: str):
    head, rest = label[1:].split("(", 1)
    digits = "".join(ch for ch in rest if ch.isdigit())
    return (int(head), int(digits))


def _record_bonds(col: _Collector, k: int, bonds, symbols: np.ndarray,
                  cn: np.ndarray, cations, anions, formers, bridging,
                  angle_edges, length_edges, available) -> set[str]:
    """Every bond-based descriptor of one frame under one bond definition.

    ``available``: the (cation, anion) pairs whose bonds are defined (None:
    every pair, the bond-valence definition). A descriptor needing a pair
    without a definition is not recorded; its name is returned.
    """
    skipped: set[str] = set()

    def have(pairs) -> bool:
        return available is None or set(pairs) <= available

    def partners(element):
        return ([(element, a) for a in anions] if element in cations
                else [(c, element) for c in cations])

    for element in sorted(cations + anions):
        name = f"CN {element}"
        if not have(partners(element)):
            skipped.add(name)
            continue
        col.counts(name, k, cn_counts(cn, symbols, element), kind="fraction")
        col.scalar(f"mean CN {element}", k,
                   math.fsum(cn[symbols == element].tolist())
                   / int((symbols == element).sum()), "1")
        if element == "B":
            col.counts("N4 (B3, B4, other)", k, boron_n4(cn, symbols),
                       kind="fraction", keys=("B3", "B4", "other"))
        if element == "Al":
            col.counts("Al CN (4, 5, 6, other)", k, aluminium_cn(cn, symbols),
                       kind="fraction", keys=("Al4", "Al5", "Al6", "other"))

    if formers is not None:
        present = [f for f in sorted(formers) if f in cations]
        for anion in anions:
            name = f"{anion} speciation"
            if have([(t, anion) for t in present]):
                col.counts(name, k, anion_speciation(bonds, symbols, formers,
                                                     anion),
                           kind="fraction", keys=SPECIATION_KEYS)
            else:
                skipped.add(name)
        if present and have([(t, a) for t in present for a in bridging]):
            qn = qn_counts(bonds, symbols, formers, bridging)
            for element in present:
                col.counts(f"Qn {element}", k, qn[element], kind="fraction",
                           keys="range")
            for (element, other), counts in qn_mx_counts(
                    bonds, symbols, formers, bridging).items():
                col.counts(f"Qn(m{other}) {element}", k, counts,
                           kind="fraction", order=_qnm_order)
            for element, value in connectivity(bonds, symbols, formers,
                                               bridging).items():
                col.scalar(f"connectivity {element}", k, value, "1")
        elif present:
            skipped.update(f"Qn {element}" for element in present)

    for anion in anions:
        if not have([(c, anion) for c in cations]):
            skipped.update({f"{anion} linkages", f"{anion} environments"})
            continue
        col.counts(f"{anion} linkages", k,
                   linkage_counts(bonds, symbols, anion), kind="count")
        col.counts(f"{anion} environments", k,
                   anion_environments(bonds, symbols, anion), kind="fraction")

    for element in sorted(cations + anions):
        if not have(partners(element)):
            skipped.add(f"angles at {element}")
            continue
        sides = anions if element in cations else cations
        found = bond_angles_deg(bonds, symbols, element)
        for i, x in enumerate(sides):
            for y in sides[i:]:
                key = (x, element, y)
                col.binned(f"angle {x}-{element}-{y}", k,
                           found.get(key, np.zeros(0)), angle_edges, "deg")

    lengths = bond_lengths_ang(bonds, symbols)
    for c in cations:
        for a in anions:
            if have([(c, a)]):
                col.binned(f"bond length {c}-{a}", k,
                           lengths.get((c, a), np.zeros(0)), length_edges, "Å")
    return skipped


@dataclass(frozen=True, eq=False)
class GlassResult:
    """Frame-averaged descriptors with their spread, and their provenance.

    ``bv`` and ``distance`` map descriptor names (each container's own
    ``name``) to md_stats containers under the two bond definitions;
    ``comparison`` holds, per element, the cross-table of the two CN on the
    same atoms. ``rdf`` and ``running_cn`` are keyed by ordered element pair;
    ``minima`` by ordered pair as well (the same minimum both ways; a pair
    whose cutoff the user gave holds that cutoff with source ``'user'``, and
    the rule's own point in its notes); ``cutoffs_ang`` / ``cutoff_sources``
    by (cation, anion).
    """

    provenance: Provenance
    composition: ModelComposition
    density: Scalar
    rdf: dict
    running_cn: dict
    minima: dict
    cutoffs_ang: dict
    cutoff_sources: dict
    bv: dict
    distance: dict
    comparison: dict
    notes: tuple[str, ...]

    def results(self) -> Iterator[tuple[str, str, object]]:
        """(section, name, container) for every result, in a stable order."""
        yield "composition", "density", self.density
        seen: set[int] = set()
        for section in ("rdf", "running_cn", "bv", "distance", "comparison"):
            for item in getattr(self, section).values():
                if id(item) in seen:            # g_ab and g_ba are one Series
                    continue
                seen.add(id(item))
                yield section, item.name, item


def _chosen_frames(trajectory: Trajectory, frames) -> list[int]:
    if frames is None or isinstance(frames, slice):
        every = range(trajectory.n_frames)
        out = list(every if frames is None else every[frames])
        if not out:
            raise ValueError(
                f"frames selects no frame (the trajectory holds "
                f"{trajectory.n_frames}; frames = {frames!r})")
        return out
    if isinstance(frames, (str, bytes)):
        raise ValueError("frames needs a slice or a sequence of frame indices")
    out = []
    for k in frames:
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int,
                                                                 np.integer)):
            raise ValueError(f"frame index {k!r} is not an integer")
        if not 0 <= int(k) < trajectory.n_frames:
            raise ValueError(f"frame {k} is outside 0 .. "
                             f"{trajectory.n_frames - 1}")
        out.append(int(k))
    if len(set(out)) != len(out):
        raise ValueError("frames names a frame more than once")
    if not out:
        raise ValueError("frames selects no frame")
    return out


def _tap(blocks: Iterable[bulk.PairTable], consumers) -> Iterator:
    """Pass each pair block to the consumers, then on to the valence table."""
    for block in blocks:
        for consumer in consumers:
            consumer.add(block)
        yield block


def _grouped(notes_by_text: dict[str, list[int]], n_used: int) -> list[str]:
    out = []
    for text, frames in notes_by_text.items():
        where = ("every frame" if len(frames) == n_used else
                 f"frames {_compact(frames)}")
        out.append(f"{text} ({where})")
    return out


def analyse_trajectory(trajectory: Trajectory, ox: ModelOxidation, *,
                       formers: Collection[str] | None,
                       rdf_r_max_ang: float,
                       minimum: MinimumMethod,
                       bins: HistogramBins,
                       params: bv.ParameterSet | None = None,
                       v_bond_vu: float = bv.V_BOND_DEFAULT,
                       v_list_vu: float = bv.V_LIST_DEFAULT,
                       frames: slice | Sequence[int] | None = None,
                       rdf_dr_ang: float = RDF_DR_ANG,
                       cutoffs_ang: Mapping[tuple[str, str], float] | None
                       = None,
                       bridging_anions: Collection[str] | None = None,
                       oxide_basis: Mapping[str, str] | None = None,
                       progress: Callable[[int, int], None] | None = None,
                       cancelled: Callable[[], bool] | None = None,
                       pair_search: Callable[[int, Frame, float, float | None],
                                             Iterable[bulk.PairTable]] | None
                       = None,
                       on_frame: Callable[[int, Frame, bulk.ValenceTable,
                                           bulk.AtomResults, bulk.Bonds], None]
                       | None = None,
                       on_minima: Callable[[dict, dict, dict], None] | None
                       = None,
                       on_distance_frame: Callable[[int, Frame, DistanceBonds],
                                                   None] | None = None
                       ) -> GlassResult:
    """Every glass descriptor on every chosen frame, averaged with its spread.

    ``formers`` has no default: a set, or None, which leaves out every
    former-dependent result with a note. ``minimum`` and ``bins`` have no
    default either; each minimum is read from the frame-averaged g(r) with
    the Poisson errors of the pairs counted over the frames used, against
    the level 1, or 1 - 1/N_a for a pair of one element (module docstring),
    and each automatic cutoff's source states its floor and the range of
    N_ab(r) across it. A smoothing kernel wider than the g(r) grid that
    ``rdf_r_max_ang`` and ``rdf_dr_ang`` give is refused before any frame is
    read. ``cutoffs_ang`` overrides the automatic first minimum of
    any cation-anion pair; when it covers every pair the analysis is one pass,
    otherwise pass 2 searches each frame again at the cutoffs (module
    docstring). ``bridging_anions`` None counts every anion of the model as a
    possible bridge, stated in the provenance. ``progress(done, total)`` is
    called after each frame of each pass; ``cancelled()`` is asked before
    each frame, and a cancelled run returns what was done, with the rest
    listed as skipped (AnalysisCancelled when nothing was done). A frame the
    reader cannot load is skipped with the reason; every skip is in the
    provenance.

    The last four arguments let a caller that runs other per-frame analyses
    on the same frames (``md_analysis.analyse``) share the frame's one pair
    search with this one; all four default to None, which leaves the
    analysis as described above. ``pair_search(k, frame, r_ang,
    vectors_within_ang)`` replaces ``bulk.iter_pairs`` in both passes: it
    returns the blocks of one search of that frame to at least ``r_ang``,
    with vectors to at least ``vectors_within_ang`` (None: every pair). A
    search wider than asked gives the same descriptors: the valence table is
    cut at the bond-valence radius, the g(r) grid and the distance bonds at
    their own limits (the deposits of g(r) may then sum in another order, a
    difference of the order of 1e-16 relative). ``on_frame(k, frame, table,
    results, bonds)`` is called after pass 1 of each frame analysed, with
    the frame's valence table, its results at ``v_bond_vu`` and its bonds;
    ``on_minima(minima, cutoffs_ang, cutoff_sources)`` once, when the minima
    of the frame-averaged g(r) and the distance cutoffs are known and before
    pass 2; ``on_distance_frame(k, frame, distance_bonds)`` after each
    frame's distance bonds are formed (in pass 1 when every cation-anion
    cutoff was given, in pass 2 otherwise). An exception raised by a
    callback ends the analysis.
    """
    if not isinstance(trajectory, Trajectory):
        raise ValueError(f"a Trajectory is needed, not "
                         f"{type(trajectory).__name__}")
    if not isinstance(ox, ModelOxidation):
        raise ValueError("ox needs a ModelOxidation (md_model.model_oxidation)")
    if not isinstance(minimum, MinimumMethod):
        raise ValueError("minimum needs a MinimumMethod (no default rule)")
    if not isinstance(bins, HistogramBins):
        raise ValueError("bins needs a HistogramBins (no default bin widths)")
    former_set = None if formers is None else _symbol_set(formers, "formers")
    params = params or bv.DEFAULT
    v_bond = _threshold(v_bond_vu, "v_bond_vu")
    v_list = _positive(v_list_vu, "v_list_vu")
    r_max_requested_ang = _positive(rdf_r_max_ang, "rdf_r_max_ang")
    step_ang = _positive(rdf_dr_ang, "rdf_dr_ang")
    chosen = _chosen_frames(trajectory, frames)
    user_cutoffs_raw = {} if cutoffs_ang is None else dict(cutoffs_ang)
    if minimum.smooth_sigma_ang is not None:
        # the kernel against the grid the arguments ask for, before any frame
        # is read (so the refusal does not depend on which frame comes
        # first); a grid the box cuts shorter gives a reason per pair
        requested = np.arange(step_ang, r_max_requested_ang + 0.5 * step_ang,
                              step_ang).size
        width = 2 * _kernel_half(step_ang, minimum.smooth_sigma_ang) + 1
        if width > requested:
            raise ValueError(
                f"smooth_sigma_ang {minimum.smooth_sigma_ang!r} Å spans "
                f"{width} grid points, more than the {requested} that "
                f"rdf_r_max_ang {r_max_requested_ang:.6g} Å and rdf_dr_ang "
                f"{step_ang:.6g} Å give; raise rdf_r_max_ang or lower "
                "smooth_sigma_ang")

    notes: list[str] = []
    skipped: dict[int, str] = {}
    used: list[int] = []
    col_bv, col_dist, col_rdf, col_cmp = (_Collector(), _Collector(),
                                         _Collector(), _Collector())
    bv_cn: dict[int, np.ndarray] = {}
    densities: dict[int, float] = {}
    half_widths: dict[int, float] = {}
    note_frames: dict[str, list[int]] = {}
    missing_pairs: dict[str, int] = {}
    estimated_pairs: dict[str, int] = {}
    zero_bonded: dict[str, list[int]] = {}
    over_limit: dict[str, list[int]] = {}
    dist_skipped: set[str] = set()
    setup: dict | None = None
    volumes: dict[int, float] = {}
    # two passes until the first frame read shows that every cation-anion
    # cutoff was given; total only ever shrinks, so done / total never falls
    done, total = 0, 2 * len(chosen)
    last: tuple[int, Frame] | None = None
    was_cancelled = False

    def report() -> None:
        if progress is not None:
            progress(done, total)

    def note_once(text: str, k: int) -> None:
        frames_seen = note_frames.setdefault(text, [])
        if not frames_seen or frames_seen[-1] != k:
            frames_seen.append(k)

    for position, k in enumerate(chosen):
        if cancelled is not None and cancelled():
            for rest in chosen[position:]:
                skipped[rest] = "not analysed: the analysis was cancelled"
            was_cancelled = True
            break
        try:
            frame = trajectory.frame(k)
        except FrameError as error:
            skipped[k] = f"could not be read: {error}"
            done += 1
            report()
            continue
        if setup is None:
            setup = _setup(frame, ox, former_set, params, v_list,
                           r_max_requested_ang, step_ang, user_cutoffs_raw,
                           bridging_anions, bins, notes)
            if setup["one_pass"]:
                total = len(chosen)
        for text in frame.notes:
            note_once(text, k)
        symbols = frame.elements
        ox_atom = setup["ox_atom"]
        cations, anions = setup["cations"], setup["anions"]

        rdf_acc = _RdfAccumulator(frame, setup["grid"], step_ang)
        consumers: list = [rdf_acc]
        dist_acc = None
        if setup["one_pass"]:
            dist_acc = _DistanceAccumulator(frame, ox_atom,
                                            setup["user_cutoffs"])
            consumers.append(dist_acc)
        if pair_search is None:
            blocks = bulk.iter_pairs(frame, setup["r_pairs"],
                                     vectors_within_ang=setup["vec_radius"])
        else:
            blocks = pair_search(k, frame, setup["r_pairs"],
                                 setup["vec_radius"])
        table = bulk.valence_table(frame, _tap(blocks, consumers), ox_atom,
                                   params, v_list_vu=v_list,
                                   r_search_ang=setup["r_bv"])
        results = bulk.at_threshold(table, v_bond)
        bonds = bulk.bonds_at(table, v_bond)
        for text in table.notes + results.notes:
            note_once(text, k)
        for label, count in table.missing_pairs.items():
            missing_pairs[label] = missing_pairs.get(label, 0) + count
        for label, count in table.estimated_pairs.items():
            estimated_pairs[label] = estimated_pairs.get(label, 0) + count

        partials = rdf_acc.result()
        for (a, b), p in partials.items():
            if a <= b:
                col_rdf.series(f"g {a}-{b}", k, p.g, setup["grid"], "r_ang",
                               "Å", "1")
            col_rdf.series(f"N {b} around {a}", k, p.n_cum, setup["grid"],
                           "r_ang", "Å", "atoms")
            for text in p.notes:
                note_once(text, k)
        half_widths[k] = float(frame.perpendicular_widths_ang.min()) / 2.0
        volumes[k] = float(frame.volume_ang3)

        cn = np.asarray(results.cn)
        _record_bonds(col_bv, k, bonds, symbols, cn, cations, anions,
                      former_set, setup["bridging"], setup["angle_edges"],
                      setup["length_edges_bv"], None)
        for element in sorted(cations + anions):
            rows = symbols == element
            # the triangle inequality bounds phi by 1; an atom with one bond
            # has phi = |u| = 1, which the division can round to 1 + 2e-16
            # (and a rigid move to 1 - 1e-16): the excess is rounding, so it
            # is clipped before binning and the atom stays in the last bin
            col_bv.binned(f"phi {element}", k,
                          np.minimum(results.phi[rows], _PHI_RANGE),
                          setup["phi_edges"], "1")
            col_bv.samples(f"plateau width {element}", k,
                           results.plateau_decades[rows], "decades",
                           bins.plateau_decades)
        _tally_extremes(cn, symbols, anions, k, zero_bonded, over_limit)
        bv_cn[k] = cn.astype(np.int32)

        if dist_acc is not None:
            dbonds = dist_acc.result(cations, anions)
            dist_skipped |= _record_distance(
                col_dist, col_cmp, k, dbonds, symbols, cations, anions,
                former_set, setup, bv_cn[k], setup["length_edges_user"])
        densities[k] = density_g_per_cm3(frame)
        if on_frame is not None:
            on_frame(k, frame, table, results, bonds)
        if dist_acc is not None and on_distance_frame is not None:
            on_distance_frame(k, frame, dbonds)
        used.append(k)
        last = (k, frame)
        done += 1
        report()

    if not used:
        if was_cancelled:
            raise AnalysisCancelled("cancelled before any frame was analysed")
        raise ValueError("no frame could be analysed: " + "; ".join(
            f"frame {k}: {reason}" for k, reason in sorted(skipped.items())))
    if was_cancelled:
        notes.append(f"cancelled after {len(used)} of {len(chosen)} chosen "
                     "frames; the others are listed as skipped")

    # -- g(r) averaged, its minima, the cutoffs --------------------------------
    empty: list[str] = []
    rdf_all = col_rdf.build(empty)
    grid = setup["grid"]
    # the grid notes come from every frame used, never from the first one
    # read, so they do not depend on the order the frames arrive in
    widths = [half_widths[k] for k in used]
    limit = min(widths)
    keep = grid <= limit
    if setup["grid_requested"][-1] > limit:
        notes.append(
            f"the g(r) grid stops at {limit:.6g} Å, half the smallest "
            "perpendicular width of the box"
            + (" over the frames used" if max(widths) != limit else "")
            + f", below the {r_max_requested_ang:.6g} Å asked for: beyond it "
            "a shell around an atom overlaps its own periodic images")
    if max(widths) != limit:
        notes.append(
            f"the box changes between frames: half its smallest perpendicular "
            f"width runs from {limit:.6g} to {max(widths):.6g} Å over the "
            "frames used")
    if not keep.all():
        rdf_all = {name: Series(s.name, s.axis[keep], s.axis_name, s.axis_unit,
                                s.value_unit, s.per_frame[:, keep],
                                frames=s.frames, notes=s.notes)
                   for name, s in rdf_all.items()}
    species = sorted(setup["cations"] + setup["anions"])
    counts = setup["counts"]
    # the mean volume: exact for the counting error at the uncorrelated level
    # when the box changes, and the volume itself when it does not
    volume_mean = math.fsum(volumes[k] for k in used) / len(used)
    rdf, running, minima = {}, {}, {}
    for i, a in enumerate(species):
        for b in species[i:]:
            series = rdf_all[f"g {a}-{b}"]
            rdf[(a, b)] = series
            if a != b:
                rdf[(b, a)] = series
            found = _pair_minimum(series, a, b, counts, volume_mean, step_ang,
                                  len(used), minimum)
            minima[(a, b)] = found
            if a != b:                       # g_ab is g_ba: one minimum
                minima[(b, a)] = dataclasses.replace(found, pair=(b, a))
    for a in species:
        for b in species:
            running[(a, b)] = rdf_all[f"N {b} around {a}"]

    cutoffs = dict(setup["user_cutoffs"])
    sources = {pair: "user" for pair in cutoffs}
    unresolved = []
    automatic = []
    for c in setup["cations"]:
        for a in setup["anions"]:
            found = minima[(c, a)]
            if (c, a) in cutoffs:
                given = cutoffs[(c, a)]
                seen = (f"the rule's own point is {found.r_ang:.6g} Å"
                        if found.found else f"the rule finds none "
                        f"({found.reason})")
                user = RdfMinimum(
                    (c, a), given, None, None, "given by the user", "user",
                    level=found.level,
                    notes=(f"the {c}-{a} cutoff {given:.6g} Å was given by "
                           f"the user; {seen}",))
                minima[(c, a)] = user
                minima[(a, c)] = dataclasses.replace(user, pair=(a, c))
                continue
            if found.found:
                cutoffs[(c, a)] = found.r_ang
                sources[(c, a)] = _auto_source(found, running[(c, a)].mean)
                notes.extend(found.notes)
                automatic.append(f"{c}-{a}")
            else:
                unresolved.append(f"{c}-{a} ({found.reason})")
    if automatic and len(used) == 1:
        notes.append("the automatic cutoffs (" + ", ".join(automatic)
                     + ") rest on the pair counts of one frame")
    if unresolved:
        notes.append("no distance cutoff for " + "; ".join(unresolved)
                     + "; the distance-definition descriptors that need one "
                     "are not reported")
    if on_minima is not None:
        on_minima(dict(minima), dict(cutoffs), dict(sources))

    # -- pass 2: the distance definition, when its cutoffs came from g(r) -----
    passes = 1
    if not setup["one_pass"]:
        if cutoffs:
            passes = 2
            max_cut_ang = max(cutoffs.values())
            # a distance bond is no longer than the largest cutoff: the bond
            # length edges stop there, whatever frame was read first
            length_edges = _fixed_edges(bins.length_ang, max_cut_ang)
            order = [last[0]] + [k for k in used if k != last[0]]
            dist_done: list[int] = []
            for k in order:
                if cancelled is not None and cancelled():
                    rest = [x for x in order if x not in dist_done]
                    notes.append(
                        "cancelled in pass 2: the distance definition covers "
                        f"frames {_compact(dist_done) if dist_done else 'none'}"
                        f" and leaves out frames {_compact(rest)}")
                    break
                try:
                    frame = last[1] if k == last[0] else trajectory.frame(k)
                except FrameError as error:
                    notes.append(f"frame {k} was read in pass 1 but not in pass "
                                 f"2 ({error}); the distance definition leaves "
                                 "it out")
                    done += 1
                    report()
                    continue
                acc = _DistanceAccumulator(frame, setup["ox_atom"], cutoffs)
                second = (bulk.iter_pairs(frame, max_cut_ang)
                          if pair_search is None else
                          pair_search(k, frame, max_cut_ang, None))
                for block in second:
                    acc.add(block)
                dbonds = acc.result(setup["cations"], setup["anions"])
                dist_skipped |= _record_distance(
                    col_dist, col_cmp, k, dbonds, frame.elements,
                    setup["cations"], setup["anions"], former_set, setup,
                    bv_cn[k], length_edges)
                if on_distance_frame is not None:
                    on_distance_frame(k, frame, dbonds)
                dist_done.append(k)
                done += 1
                report()
        else:
            notes.append("no cation-anion pair has a distance cutoff, so the "
                         "distance definition is not computed")
    if progress is not None and done < total:
        done = total
        report()

    # -- containers, notes, provenance -----------------------------------------
    bv_results = col_bv.build(empty)
    distance_results = col_dist.build(empty)
    comparison = col_cmp.build(empty)
    if empty:
        notes.append("histograms and tables with no entry in any frame are not "
                     "listed: " + ", ".join(sorted(set(empty))))
    if dist_skipped:
        notes.append("distance definition, not reported for want of a cutoff: "
                     + ", ".join(sorted(dist_skipped)))
    for anion, frames_seen in zero_bonded.items():
        notes.append(f"{anion} atoms bonded to no cation at v_bond = {v_bond:g} "
                     f"v.u. in frames {_compact(frames_seen)}")
    for element, frames_seen in over_limit.items():
        notes.append(f"{element} with CN above {CN_NOTE_LIMITS[element]} at "
                     f"v_bond = {v_bond:g} v.u. in frames "
                     f"{_compact(frames_seen)} (the MD prompt names this as "
                     "physically impossible)")
    for anion, frames_seen in setup["dist_zero"].items():
        notes.append(f"{anion} atoms bonded to no cation within the distance "
                     f"cutoffs in frames {_compact(frames_seen)}")
    for element, frames_seen in setup["dist_over"].items():
        notes.append(f"{element} with CN above {CN_NOTE_LIMITS[element]} within "
                     f"the distance cutoffs in frames {_compact(frames_seen)} "
                     "(the MD prompt names this as physically impossible)")
    notes.extend(_grouped(note_frames, len(used)))
    if pair_search is not None:
        notes.append("the pair blocks of each frame came from the caller's "
                     "pair_search, a search of the frame shared with other "
                     "analyses, to at least the radii this analysis asked for")

    model = composition(last[1], ox, oxide_basis)
    density = Scalar("density", "g/cm^3", [densities[k] for k in used],
                     frames=used)
    method_parameters = {
        "rdf_dr_ang": step_ang,
        "rdf_r_max_ang requested": r_max_requested_ang,
        "rdf_r_max_ang used": float(grid[keep][-1]),
        "minimum rule": minimum.rule,
        "minimum smooth_sigma_ang": minimum.smooth_sigma_ang,
        "minimum flat_rule": minimum.flat_rule,
        "minimum margin_std_errors": minimum.margin_std_errors,
        "angle bin (deg)": bins.angle_deg, "phi bin": bins.phi,
        "plateau bin (decades)": bins.plateau_decades,
        "bond length bin (Å)": bins.length_ang,
        "bridging anions": tuple(sorted(setup["bridging"])),
        "pair searches per frame": passes,
    }
    provenance = Provenance.from_trajectory(
        trajectory, frames_used=used, frames_skipped=skipped, params=params,
        notes=tuple(notes) + tuple(model.notes), ox=ox, v_bond_vu=v_bond,
        v_list_vu=v_list, r_search_ang=setup["r_bv"], formers=former_set,
        cutoffs_ang=cutoffs, cutoff_sources=sources,
        method_parameters=method_parameters, estimated_pairs=estimated_pairs,
        missing_pairs=missing_pairs)
    return GlassResult(
        provenance=provenance, composition=model, density=density, rdf=rdf,
        running_cn=running, minima=minima, cutoffs_ang=dict(sorted(
            cutoffs.items())), cutoff_sources=dict(sorted(sources.items())),
        bv=bv_results, distance=distance_results, comparison=comparison,
        notes=tuple(notes))


def _setup(frame: Frame, ox: ModelOxidation, former_set, params, v_list: float,
           r_max_requested_ang: float, step_ang: float, user_cutoffs_raw,
           bridging_anions,
           bins: HistogramBins, notes: list[str]) -> dict:
    """What every frame shares, fixed on the first frame read."""
    symbols = frame.elements
    ox_atom = ox.per_atom(symbols)
    cations, anions = _roles(symbols, ox_atom)
    notes.extend(ox.notes)
    if former_set is None:
        notes.append("network formers not given: no speciation, Qn, Qn(mX) or "
                     "connectivity is reported (no former set is assumed)")
    else:
        bad = sorted(f for f in former_set if f in anions)
        if bad:
            raise ValueError(f"{', '.join(bad)} are anions of this model and "
                             "cannot be network formers (a former is a "
                             "cation, oxidation state 0 or more)")
        absent = sorted(f for f in former_set if f not in cations)
        if absent:
            notes.append(f"former(s) {', '.join(absent)} are not in the model")
        if not former_set:
            notes.append("an empty former set was given: every anion is free "
                         "and no former has a Qn")
    if bridging_anions is None:
        bridging = frozenset(anions)
        notes.append("bridging anions: every anion of the model ("
                     + ", ".join(anions) + "), since none were named")
    else:
        bridging = _symbol_set(bridging_anions, "bridging_anions")
        stray = sorted(b for b in bridging if b not in anions)
        if stray:
            raise ValueError(f"bridging anion(s) {', '.join(stray)} are not "
                             "anions of this model")
        if not bridging:
            notes.append("an empty bridging-anion set was given: no anion "
                         "bridges, so every former has n = 0 (Q0) and the "
                         "connectivity is 0, whatever the anion speciation")
    user_cutoffs, more = _cation_anion_cutoffs(user_cutoffs_raw, cations,
                                               anions)
    notes.extend(more)
    one_pass = all((c, a) in user_cutoffs for c in cations for a in anions)
    r_bv = bulk.search_radius_ang(symbols, ox_atom, params, v_list)
    # the grid notes are written once every frame is in (analyse_trajectory)
    grid, _ = rdf_grid_ang(frame, r_max_requested_ang, step_ang)
    if grid.size < 3:
        raise ValueError(
            f"rdf_r_max_ang {r_max_requested_ang:.6g} Å with rdf_dr_ang "
            f"{step_ang:.6g} Å gives a g(r) grid of {grid.size} points on "
            "this box; a first minimum needs at least three, so raise "
            "rdf_r_max_ang or lower rdf_dr_ang")
    rdf_need = float(grid[-1] + step_ang)
    user_max = max(user_cutoffs.values()) if user_cutoffs else 0.0
    # bond lengths: a bond-valence bond is a listed contact, within r_bv; a
    # distance bond is within its cutoff. Neither depends on the frame, so
    # the edges do not depend on which frame is read first.
    return {
        "ox_atom": ox_atom, "cations": cations, "anions": anions,
        "counts": dict(frame.composition),
        "bridging": bridging, "user_cutoffs": user_cutoffs,
        "one_pass": one_pass, "r_bv": r_bv, "grid": grid,
        "grid_requested": np.arange(step_ang, r_max_requested_ang
                                    + 0.5 * step_ang, step_ang),
        "r_pairs": max(r_bv, rdf_need, user_max),
        "vec_radius": max(r_bv, user_max),
        "angle_edges": _fixed_edges(bins.angle_deg, _ANGLE_RANGE_DEG),
        "phi_edges": _fixed_edges(bins.phi, _PHI_RANGE),
        "length_edges_bv": _fixed_edges(bins.length_ang, r_bv),
        "length_edges_user": _fixed_edges(bins.length_ang,
                                          max(user_max, bins.length_ang)),
        "dist_zero": {}, "dist_over": {},
    }


def _pair_minimum(series: Series, a: str, b: str, counts: Mapping[str, int],
                  volume_ang3: float, step_ang: float, n_frames: int,
                  minimum: MinimumMethod) -> RdfMinimum:
    """The first minimum of one frame-averaged g_ab(r), with its counting
    errors and its uncorrelated level."""
    na, nb = int(counts[a]), int(counts[b])
    label = f"g_{a}{b}(r)"
    if a == b and na < 2:
        return RdfMinimum((a, b), None, None, None, minimum.describe(), "auto",
                          f"{label}: the model holds one {a} atom, so it has "
                          f"no {a}-{a} pair", level=0.0)
    if series.axis.size < 3:
        return RdfMinimum((a, b), None, None, None, minimum.describe(), "auto",
                          f"{label}: the g(r) grid common to the frames used "
                          f"holds {series.axis.size} point(s); a first minimum "
                          "needs at least three")
    level = (na - 1) / na if a == b else 1.0
    sigma_ang = minimum.smooth_sigma_ang
    if sigma_ang is not None:
        width = 2 * _kernel_half(step_ang, sigma_ang) + 1
        if width > series.axis.size:
            return RdfMinimum(
                (a, b), None, None, None, minimum.describe(), "auto",
                f"{label}: the smoothing sigma of {sigma_ang!r} Å spans {width} "
                f"grid points, more than the {series.axis.size} of the g(r) "
                "grid common to the frames used, which stops at half the "
                "smallest perpendicular box width", level=level)
    per_unit = pair_counts_per_unit_g(series.axis, step_ang, na, nb,
                                      volume_ang3, same_element=a == b,
                                      n_frames=n_frames)
    return first_minimum(series.axis, series.mean, minimum, pair=(a, b),
                         level=level, pairs_per_unit_g=per_unit)


def _auto_source(found: RdfMinimum, n_cum) -> str:
    """The cutoff source of an automatic first minimum: its rule, its floor,
    the running CN across the floor, and the depth of its lowest point."""
    lo, hi = found.floor_index
    a, b = found.pair
    depth = ""
    if found.depth_std_errors is not None:
        value = found.depth_std_errors
        depth = (f"; its lowest point lies {abs(value):.3g} standard errors "
                 f"{'below' if value >= 0 else 'above'} {found.level:.6g}")
    return (f"first minimum of the frame-averaged g_{a}{b}(r), "
            f"{found.method}; its floor runs {found.floor_r_ang[0]:.6g}"
            f"-{found.floor_r_ang[1]:.6g} Å, across which N_{a}{b}(r) "
            f"runs {n_cum[lo]:.6g}-{n_cum[hi]:.6g}{depth}")


def _tally_extremes(cn, symbols, anions, k, zero_bonded, over_limit,
                    limits: Mapping[str, int] = CN_NOTE_LIMITS) -> None:
    """Record frame k for each anion with an atom bonded to no cation and
    each element of ``limits`` with an atom above its limit."""
    for anion in anions:
        if ((cn == 0) & (symbols == anion)).any():
            zero_bonded.setdefault(anion, []).append(k)
    for element, limit in limits.items():
        if ((cn > limit) & (symbols == element)).any():
            over_limit.setdefault(element, []).append(k)


def _record_distance(col_dist: _Collector, col_cmp: _Collector, k: int,
                     dbonds: DistanceBonds, symbols, cations, anions,
                     former_set, setup: dict, cn_bv: np.ndarray,
                     length_edges: np.ndarray) -> set[str]:
    """The distance-definition descriptors of one frame, and the comparison."""
    available = set(dbonds.cutoffs_ang)
    cn_dist = bond_cn(dbonds, symbols.shape[0])
    skipped = _record_bonds(col_dist, k, dbonds, symbols, cn_dist, cations,
                            anions, former_set, setup["bridging"],
                            setup["angle_edges"], length_edges, available)
    for element in sorted(cations + anions):
        if f"CN {element}" in skipped:
            continue
        rows = symbols == element
        pairs, counts = np.unique(np.stack([cn_bv[rows], cn_dist[rows]],
                                           axis=1), axis=0, return_counts=True)
        col_cmp.counts(f"CN {element} (BV, distance)", k,
                       {(int(a), int(b)): int(c) for (a, b), c in
                        zip(pairs, counts)}, kind="fraction")
        col_cmp.scalar(f"CN {element} BV-distance differences", k,
                       int((cn_bv[rows] != cn_dist[rows]).sum()), "atoms")
    # the same notes as the bond-valence definition, for the elements whose
    # distance CN is complete (every counter-ion pair has a cutoff)
    complete = [e for e in anions if f"CN {e}" not in skipped]
    limits = {e: v for e, v in CN_NOTE_LIMITS.items() if f"CN {e}" not in skipped}
    _tally_extremes(cn_dist, symbols, complete, k, setup["dist_zero"],
                    setup["dist_over"], limits)
    return skipped
