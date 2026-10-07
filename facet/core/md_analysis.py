"""Every MD analysis of a model in one run, with one pair search per frame.

The analysis modules of the MD path each measure one family of descriptors
on a trajectory: :mod:`.glass` (coordination, Q^n, speciation, angles,
partial g(r)), :mod:`.md_scattering` (S(Q), G(r)), :mod:`.md_network` (rings,
coordination sequences, polyhedral sharing, percolation, chemical order),
:mod:`.md_order` (bond-orientational order, polyhedron distortion, Voronoi
cells, voids), :mod:`.md_spectroscopy` (NMR shifts from correlations, EXAFS
shells) and :mod:`.md_dynamics` (diffusion, vibrations, conduction,
lifetimes). Each of them, called on its own, reads every frame and searches
it for neighbour pairs. A student who wants all of them would read and
search every frame six times. This module runs them together:
:func:`analyse` takes a :class:`~.md_model.Trajectory` and an
:class:`AnalysisRequest` and returns a :class:`ModelResult` holding every
requested result with its provenance, and :mod:`.md_export` writes it out.

ONE PAIR SEARCH PER FRAME
-------------------------
Every per-frame analysis that needs neighbour pairs takes them from one
``bulk.iter_pairs`` call per frame, made here at the largest radius any of
them needs. The search streams its blocks of centre atoms (bulk.py,
MEMORY); as each block passes, the pairs each other analysis needs are kept
as a restricted copy: the pairs with d <= its own radius, with vectors only
for an analysis that reads them. A restricted copy is exactly what a search
of that frame at that radius gives (the same pairs, the same distances and
vectors, formed from the same fractions by the same expression,
``bulk._pair_geometry``), so each analysis receives what its own search
would have found, in another block order. What the order can change is the
last bit of a floating-point sum (a g(r) deposit summed block by block).
``tests/test_md_analysis.py`` compares the driver with each module called
directly on the same frames of a jittered quartz supercell (integer
descriptors exactly, the rest to 1e-12); on five frames of the 3 000-atom
Na2O-3SiO2 glass, glass, scattering and EXAFS through the driver against
the three modules on their own searches gave every count, histogram, N(r),
first minimum, cutoff, shell and scattering curve identical, and the glass
g(r) within 1.3e-14 (2.3e-15 relative), with 10 searches against 25
(measured 2026-10-07). The glass analysis keeps its own frame loop and
calls back into this module (``glass.analyse_trajectory``'s
``pair_search``, ``on_frame``, ``on_minima`` and ``on_distance_frame``
arguments), so its valence table, results and bonds are the ones every
bond-based analysis here reads; the scattering module is given each frame's
partials (its ``given_partials`` argument) instead of searching.

Some cutoffs exist only after every frame has been searched: the first
minimum of a frame-averaged g(r) (glass) and the first-shell limit of an
absorber-centred g(r) (EXAFS). An analysis that reads pairs within such a
cutoff therefore needs a second look at each frame, after the first pass.
That is pass 2: each frame read again and searched once more, at the largest
cutoff pass 2 reads, and shared the same way (on the Na2O-3SiO2 glass 3.4 Å
for the glass and EXAFS first shells, 4.43 Å once Warren-Cowley reads every
element pair, its Na-Si minimum being the widest; measured 2026-10-07). A
run whose cutoffs are all given by the user has one pass; the
provenance states the number of searches each frame received
(``ModelResult.searches``), and ``tests/test_md_analysis.py`` counts them.
The trajectory analyses of :mod:`.md_dynamics` read the unwrapped positions
of the chosen frames once more (``md_dynamics.collect_tracks``); they search
nothing, except the distinct van Hove function, which searches the frames it
builds from two times (its own docstring). The FEFF export
(``md_spectroscopy.export_feff_inputs``) searches each frame that holds a
drawn absorber once more, to the cluster radius; it writes files rather
than measuring, and its note says so.

What one search saves depends on which analysis needs the widest one. The
scattering grid runs to half the box by default, and on a 3 000-atom glass
frame (34.6 Å box) a search to 17.4 Å took 2.0 s against 0.12 s to the 6 Å
of the bond valences and 0.03 s to a 3.4 Å pass-2 cutoff (Na2O-3SiO2, one
frame, minimum of 3, measured 2026-10-07): with the scattering in the run,
that one search is most of the cost either way, and the narrower searches
the driver saves are a tenth of it. Without it, the searches the driver
saves are most of the searching.

WHAT HAS NO DEFAULT, AND WHAT HAS ONE
-------------------------------------
Physical inputs have none: the network formers, the MD timestep or frame
interval, the temperature, the ionic charges, the NMR correlation and its
reference, the atom radii and probe of a void analysis, the absorber of an
EXAFS analysis, the measured data a comparison reads. Choices that define
what is measured and have no measured basis here have none either: the ring
criterion and its largest size, the number of coordination shells, the
MSD fit window, the r-window of the sine transform. A request that lacks one
is refused before any frame is read, with every missing input named at once
(:func:`missing_inputs`).

So is a request that gives a value no analysis can take
(:func:`invalid_inputs`): a bin width or grid step of 0 (a division by it
would end the run, some of them only once every frame had been analysed), a
name outside a module's own list (a ring criterion, a radiation, a
lineshape), a window whose ends are reversed, a measured file that does not
exist, both timestep_fs and frame_interval_ps, or v_bond below v_list. That
last one is a contradiction rather than a range: a contact below v_list is
never tabulated, so a CN at a v_bond below it is the CN at v_list, and on
the Na2O-3SiO2 glass v_bond 0.01 with v_list 0.02 reported CN Na 5.59 under
the label 0.01 v.u., where the CN at 0.01 v.u. is 6.65 (measured
2026-10-07). Against the model (the first chosen frame) the run refuses an
element an option names that the model does not hold, network formers none
of which are in the model (a typo would otherwise give 100 % free O), a
scattering r_max beyond half the box, a frame selection that selects
nothing, and a time-axis analysis without a time axis, velocities or two
frames. Named formers of which only some are in the model, and a
composition that is not neutral with the oxidation states used, are model
notes (:attr:`ModelResult.model_notes`), carried into every export header.

Method choices that only set a resolution or a presentation have defaults
here, each named in its options class, stated in the provenance of the
result, and overridable: bin widths, grid steps, the first-minimum rule.
The first-minimum rule defaults to the 'valley' rule at a margin of 2
standard errors with the midpoint of the floor: on the four 3 000-atom
melt-quenched glasses of the MD verification, that rule put every cation-
anion cutoff inside the floor that contains LAMMPS's own g(r) minimum
(glass.py, THE FIRST MINIMUM). The g(r) radius defaults to the bond-valence
search radius, so the g(r) asks for no wider search than the bond valences
do; the scattering r grid defaults to the largest one the box holds, because
the sine transform resolves Q to about pi / r_max. When the box changes
between frames (NPT), that is the grid the smallest box of the frames used
holds: each frame's partials are computed to what its own box holds (at
most the first frame's grid, which sets the shared search) and every frame
is cut to the shortest, as glass cuts its g(r), so the average does not
depend on the order of the frames. Taking the first frame's grid instead
refused every smaller frame: on four frames of the Na2O-3SiO2 glass scaled
by 1, 0.9995, 1.0005 and 0.999, the forward order averaged S(Q) over 2
frames and the reversed order over 4, a difference of 7.2e-3 (measured
2026-10-07). A cut grid is exact: the grid points below the cut hold the
same deposits whatever grid continues beyond them
(``tests/test_md_analysis.py``).

PROGRESS AND CANCELLING
-----------------------
``progress(done, total, stage)`` is called after each frame of each pass and
after each analysis that runs on the whole trajectory; ``total`` never
grows. A pass-1 frame counts :data:`PROGRESS_UNITS_PASS1` units and a pass-2
frame :data:`PROGRESS_UNITS_PASS2`, each later step one: pass 1 runs every
per-frame analysis and pass 2 only reads the narrower pairs. With every
analysis on the 20 frames of the Na-borosilicate glass a pass-1 frame took
8.4 s and a pass-2 frame 0.21 s; with glass alone on the Na2O-3SiO2 glass,
0.50 s and 0.35 s, most of it reading the frame again (measured
2026-10-07). No fixed ratio fits both; 10 to 1 is a choice of presentation
that follows the long runs, where a progress line matters (glass alone then
shows 90 % at 59 % of its 17 s), and the stage text names the pass and the
frame. When each frame counted one unit whatever its pass, the 20-frame
SiO2 run with every analysis showed 38 % at 160 s of 181 s.
``cancelled()`` is asked before each frame and between analyses, and
the ring and coordination-sequence searches ask it as they go. A cancelled
run returns what was finished: each analysis is averaged over the frames it
received, its provenance lists the others as skipped, and the notes say
that the run was cancelled. A run cancelled before any frame was analysed,
or before any analysis produced a result, raises :class:`AnalysisCancelled`.

MEMORY
------
Frames are read one at a time and their pair blocks streamed; what a run
keeps is each analysis's per-frame results until they are averaged (the
partials of each frame for the scattering, about 0.3 MB a frame for three
elements on a 1 700-point grid; the ring statistics; the Voronoi counts and
volumes without the face table; the empty-sphere radii without the
tetrahedra; each frame's bonds without their vectors for the lifetimes),
and during a frame the restricted copies of its pairs. The copy that keeps
every pair of the search (the scattering's, when its grid sets the radius)
shares the search's arrays instead of copying them. The dynamics read the
positions of every chosen frame into memory, as md_dynamics does.

TIMINGS
-------
Measured 2026-10-07 on this machine (Windows 11, i5-13420H, Python 3.11.9,
numpy 2.4.6, scipy 1.15.1), unpinned, one run per model in a fresh process,
with other sessions running: :func:`analyse` with every one of the 25
analyses on the 20 frames of the 300 K NVT trajectory of each of the four
melt-quenched glasses of the MD verification (LAMMPS dumps, 2 880-3 000
atoms in 34-35 Å boxes): the scattering grid to half the box (Lorch
r-window, neutron and X-ray, terminated at 18 Å^-1, FSDP), primitive rings
to 12 T atoms on the bridged graph, six coordination shells, Warren-Cowley
on the automatic first minima, Steinhardt l = 4 and 6, q_tet and polyhedron
shape on the bonds, Voronoi cells, empty spheres and free volume (van der
Waals radii, 0.5 Å probe, 0.5 Å grid), a test-value NMR correlation, EXAFS
about Si with automatic limits, bond lifetimes with residence times, every
dynamics analysis, both comparisons and three FEFF clusters. Each frame was
searched twice (pass 1 and pass 2), and no analysis failed::

                         SiO2    Na2O-3SiO2   Na-aluminosilicate   Na-borosilicate
    analyse, total       195 s   149 s        166 s                158 s
    shared searches       46 s    49 s         42 s                 46 s
    rings                 52 s    14 s         38 s                 27 s
    free volume           37 s    30 s         32 s                 29 s
    Voronoi cells         18 s    18 s         18 s                 18 s
    scattering            15 s    14 s         13 s                 14 s
    empty spheres          8 s     8 s          8 s                  8 s
    glass                  6 s     7 s          6 s                  7 s
    descriptors, rows     208, 66 691   277, 101 403   373, 129 011   373, 132 987
    peak commit          1.14 GB  1.09 GB      1.11 GB              1.12 GB

(7.5-9.7 s a frame; the other analyses took under 3 s each over the 20
frames; the peak commit includes the 0.38 GB the process held after its
imports.) Writing the results took 1.5-3.1 s as CSV files and 14-27 s as a
workbook (md_export.py, TIMINGS).
"""
from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import bulk, bv, glass, md_dynamics, md_network, md_order, \
    md_scattering, md_spectroscopy
from .md_model import Frame, FrameError, Trajectory, model_oxidation, \
    validate_symbol
from .md_stats import Distribution, Histogram, Provenance, Scalar, Series, \
    _compact

__all__ = [
    "ANALYSES", "ANALYSIS_SUMMARIES", "PER_FRAME_ANALYSES",
    "TRAJECTORY_ANALYSES", "POST_ANALYSES", "NO_FORMERS", "CHARGE_SOURCES",
    "RADII_NAMES", "CURVE_FUNCTIONS", "GRAPH_KINDS", "NEIGHBOUR_KINDS",
    "HISTOGRAM_BINS_MAX", "HISTOGRAM_WINDOW_PERCENTILES",
    "PROGRESS_UNITS_PASS1", "PROGRESS_UNITS_PASS2", "FORMER_READERS",
    "AnalysisCancelled", "FramesUnreadable", "MissingInput", "RequestError",
    "GlassOptions", "ScatteringOptions", "MeasuredCurve", "NetworkOptions",
    "OrderOptions", "VoidOptions", "NmrOptions", "FractionMeasurement",
    "ExafsOptions", "FeffOptions", "DynamicsOptions", "AnalysisRequest",
    "missing_inputs", "invalid_inputs", "available_analyses", "read_model",
    "Table", "AnalysisOutput", "ModelResult", "analyse",
]

# ---------------------------------------------------------------------------
# the analyses this driver runs
# ---------------------------------------------------------------------------

ANALYSIS_SUMMARIES: dict[str, str] = {
    "glass": "CN by bond valence and by distance, O speciation, Qn, Qn(mX), "
             "N4, Al CN, linkages, anion environments, bond angles and "
             "lengths, phi and plateau widths, partial g(r) and N(r) with "
             "their first minima, composition, density (glass)",
    "scattering": "partial and total S(Q), F(Q), G(r), D(r), T(r), "
                  "Bhatia-Thornton, FSDP (md_scattering)",
    "rings": "ring statistics R_C, R_N, P_N, P_max, P_min and ring sizes, "
             "King, Guttman or primitive (md_network)",
    "coordination-sequences": "N_k, nodes k bonds away (md_network)",
    "polyhedral-sharing": "corner, edge and face sharing between polyhedra "
                          "(md_network)",
    "components": "connected pieces, their dimensionality and the largest "
                  "fraction (md_network)",
    "warren-cowley": "Warren-Cowley chemical short-range order alpha_ij on "
                     "a distance graph (md_network)",
    "bond-order": "Steinhardt q_l, w_l and their averages per atom, global "
                  "Q_l (md_order)",
    "tetrahedral-order": "Errington-Debenedetti q_tet (md_order)",
    "polyhedron-shape": "Baur distortion, angle variance, quadratic "
                        "elongation, polyhedron volume, ECoN (md_order)",
    "voronoi": "Voronoi cell volume, face count and index (md_order)",
    "empty-spheres": "empty sphere of every Delaunay tetrahedron "
                     "(md_order)",
    "free-volume": "geometric, probe-centre and probe-occupiable free volume "
                   "(md_order)",
    "nmr": "isotropic shifts and spectrum from a published correlation "
           "(md_spectroscopy)",
    "exafs": "absorber-centred g(r), first-shell limits, shell cumulants "
             "N, R, sigma^2, C3, C4 (md_spectroscopy)",
    "bond-lifetimes": "continuous and intermittent bond correlation "
                      "functions, residence times (md_dynamics)",
    "msd": "mean-square displacement and diffusion coefficients "
           "(md_dynamics)",
    "self-correlations": "non-Gaussian parameter, self van Hove function, "
                         "self intermediate scattering function "
                         "(md_dynamics)",
    "distinct-van-hove": "distinct van Hove function G_d(r, t) "
                         "(md_dynamics)",
    "vacf": "velocity autocorrelation, vibrational density of states, "
            "Green-Kubo diffusion (md_dynamics)",
    "kinetic-temperature": "kinetic temperature of the frames "
                           "(md_dynamics)",
    "conductivity": "Nernst-Einstein and collective ionic conductivity, "
                    "Haven ratio (md_dynamics)",
    "nmr-comparison": "model fractions (Qn, N4, Al CN, speciation) beside "
                      "measured ones (md_spectroscopy)",
    "scattering-comparison": "a model S(Q) or G(r) against a measured curve, "
                             "R_chi (md_scattering)",
    "feff": "FEFF inputs for absorbers drawn across the frames "
            "(md_spectroscopy)",
}
ANALYSES = tuple(ANALYSIS_SUMMARIES)
PER_FRAME_ANALYSES = ("glass", "scattering", "rings", "coordination-sequences",
                      "polyhedral-sharing", "components", "warren-cowley",
                      "bond-order", "tetrahedral-order", "polyhedron-shape",
                      "voronoi", "empty-spheres", "free-volume", "nmr",
                      "exafs", "bond-lifetimes")
TRAJECTORY_ANALYSES = ("msd", "self-correlations", "distinct-van-hove", "vacf",
                       "kinetic-temperature", "conductivity")
POST_ANALYSES = ("nmr-comparison", "scattering-comparison", "feff")

# The request's formers when the user states that the model has none to name:
# the former-dependent descriptors are then left out, with a note.
NO_FORMERS = "none"
# How the ionic charges of a conductivity are stated, besides a map.
CHARGE_SOURCES = ("oxidation states", "file")
# Named radius sets of the void analyses besides a map: FACET's van der Waals
# radii (md_order.vdw_radii_ang) or radii of 0 (the circumspheres).
RADII_NAMES = ("vdw", "zero")
# The model curves a measured one can be compared with, and their axis.
CURVE_FUNCTIONS = {"S(Q)": ("s", "q_inv_ang"), "F(Q)": ("f_reduced", "q_inv_ang"),
                   "F_K(Q)": ("f_keen", "q_inv_ang"), "G(r)": ("pdf_g", "r_ang"),
                   "G_K(r)": ("keen_g", "r_ang"), "D(r)": ("keen_d", "r_ang"),
                   "T(r)": ("keen_t", "r_ang"),
                   "G(r) from S(Q)": ("pdf_g_from_sq", "r_ang")}
GRAPH_KINDS = ("bridging anion", "bond valence", "distance")
NEIGHBOUR_KINDS = ("bond valence", "distance", "voronoi")
# The most bins a histogram whose edges follow the values takes, and the
# percentiles its edges fall back on when the values need more. Choices of
# presentation, stated in a note whenever they act (_edges_over); the
# values beyond the edges are counted, never dropped.
HISTOGRAM_BINS_MAX = 2000
HISTOGRAM_WINDOW_PERCENTILES = (0.5, 99.5)
# The units one frame of each pass counts in progress(done, total): a choice
# of presentation, with its measured basis in the module docstring.
PROGRESS_UNITS_PASS1 = 10
PROGRESS_UNITS_PASS2 = 1
# The analyses that read the network formers.
FORMER_READERS = ("glass", "rings", "coordination-sequences",
                  "polyhedral-sharing", "components", "nmr")
# The analyses that need a time axis (and at least two frames).
_TIMED = TRAJECTORY_ANALYSES + ("bond-lifetimes",)


class AnalysisCancelled(Exception):
    """The run was cancelled before any frame was analysed."""


class FramesUnreadable(ValueError):
    """No chosen frame could be read: the model file, not the request."""


@dataclass(frozen=True)
class MissingInput:
    """An input a requested analysis needs and the request does not give."""

    analysis: str
    name: str                  # the request field, as 'group.field'
    why: str

    def describe(self) -> str:
        return f"{self.analysis}: {self.name}: {self.why}"


class RequestError(ValueError):
    """A request that cannot run: every missing or contradictory input."""

    def __init__(self, missing: Sequence[MissingInput]):
        self.missing = tuple(missing)
        super().__init__("the request cannot run:\n" + "\n".join(
            "  " + m.describe() for m in self.missing))


# ---------------------------------------------------------------------------
# options of each analysis; every default is a method choice and says why
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GlassOptions:
    """Method choices of the glass descriptors (``glass.analyse_trajectory``).

    ``rdf_r_max_ang`` None: the bond-valence search radius of the model
    (``bulk.search_radius_ang``, at least 6 Å), so the g(r) asks for no wider
    search than the valences. ``rdf_dr_ang``: ``glass.RDF_DR_ANG``, pdf.py's
    grid step. The minimum rule: 'valley', no smoothing, the midpoint of the
    floor, a margin of 2 standard errors (module docstring). The bin widths
    (1 degree, 0.02 in phi, 0.1 decade, 0.01 Å) set the resolution of the
    histograms only. ``cutoffs_ang`` overrides any automatic cation-anion
    cutoff (keys 'Si-O' or ('Si', 'O')); ``oxide_basis`` maps each cation to
    its oxide for the oxide mol %.
    """

    rdf_r_max_ang: float | None = None
    rdf_dr_ang: float = glass.RDF_DR_ANG
    minimum_rule: str = "valley"
    minimum_smooth_sigma_ang: float | None = None
    minimum_flat_rule: str = "midpoint"
    minimum_margin_std_errors: float = 2.0
    angle_bin_deg: float = 1.0
    phi_bin: float = 0.02
    plateau_bin_decades: float = 0.1
    length_bin_ang: float = 0.01
    cutoffs_ang: Mapping[tuple[str, str], float] | None = None
    oxide_basis: Mapping[str, str] | None = None

    def minimum(self) -> glass.MinimumMethod:
        return glass.MinimumMethod(self.minimum_rule,
                                   self.minimum_smooth_sigma_ang,
                                   self.minimum_flat_rule,
                                   self.minimum_margin_std_errors)

    def bins(self) -> glass.HistogramBins:
        return glass.HistogramBins(self.angle_bin_deg, self.phi_bin,
                                   self.plateau_bin_decades,
                                   self.length_bin_ang)


@dataclass(frozen=True)
class MeasuredCurve:
    """A measured curve to put beside a model one (scattering-comparison).

    ``function`` is one of :data:`CURVE_FUNCTIONS` and fixes the axis
    (Q in Å^-1 or r in Å); ``radiation`` one of ``md_scattering.RADIATIONS``.
    ``axis_range`` limits the points compared; ``scale`` None fits one factor
    by least squares (``md_scattering.compare_with_measured``).
    """

    path: str
    radiation: str
    function: str
    axis_range: tuple[float, float] | None = None
    scale: float | None = None


@dataclass(frozen=True)
class ScatteringOptions:
    """Inputs of the total scattering (``md_scattering.analyse_trajectory``).

    ``r_window`` and ``radiations`` are required: the module states no
    default for either. ``r_max_ang`` None takes the largest grid the
    smallest box of the frames used holds (r_max + dr within half its
    smallest perpendicular width; module docstring), since the sine
    transform resolves Q to about pi / r_max; a given ``r_max_ang`` beyond
    half the first frame's box is refused before any frame is searched. The
    Q grid runs from ``q_step_inv_ang`` to ``q_grid_max_inv_ang`` (0.02 and
    20 Å^-1 by default, a choice of range and step). ``termination_*`` and
    ``q_window`` give the terminated G(r) a measurement would show, and the
    G(r) transformed from the model's S(Q); ``termination_q_min_inv_ang``
    None starts that transform at the first Q of the grid
    (``q_step_inv_ang``), the lowest Q the S(Q) holds (0 would lie below it,
    and md_scattering refuses a range the data do not cover).
    ``fsdp_window_inv_ang`` with ``fsdp_baseline`` the first sharp
    diffraction peak. ``measured`` lists curves for scattering-comparison.
    """

    r_window: str | None = None
    radiations: tuple[str, ...] | None = None
    r_max_ang: float | None = None
    dr_ang: float = glass.RDF_DR_ANG
    q_step_inv_ang: float = 0.02
    q_grid_max_inv_ang: float = 20.0
    termination_q_max_inv_ang: float | None = None
    termination_q_min_inv_ang: float | None = None
    q_window: str | None = None
    fsdp_window_inv_ang: tuple[float, float] | None = None
    fsdp_baseline: str | None = None
    neutron_lengths_fm: Mapping[str, float] | None = None
    lengths_source: str | None = None
    measured: tuple[MeasuredCurve, ...] = ()


@dataclass(frozen=True)
class NetworkOptions:
    """Inputs of the md_network analyses.

    ``graph`` is the graph of rings, coordination sequences and components:
    'bridging anion' (formers linked through a shared anion, ring sizes in
    T atoms; the default), 'bond valence' (the CN's bonds among
    ``graph_elements``, by default the formers and the bridging anions, so
    the modifier-anion bonds do not close rings), or 'distance' (pairs within
    ``distance_cutoffs_ang``, or, when None, within the glass analysis's
    first minima of the cation-anion pairs among ``graph_elements``, which
    is then required). ``ring_criterion`` and ``ring_max_size`` are required
    for rings; ``n_shells`` for coordination sequences. Centres, ligands and
    T elements default to the formers and the model's anions, stated.
    Warren-Cowley reads a distance graph of every element pair among
    ``wc_elements`` (None: every element), with ``wc_cutoffs_ang`` or the
    glass first minima of those pairs.
    """

    graph: str = "bridging anion"
    graph_elements: tuple[str, ...] | None = None
    distance_cutoffs_ang: Mapping[tuple[str, str], float] | None = None
    ring_criterion: str | None = None
    ring_max_size: int | None = None
    ring_t_elements: tuple[str, ...] | None = None
    n_shells: int | None = None
    cseq_centres: tuple[str, ...] | None = None
    sharing_centres: tuple[str, ...] | None = None
    sharing_ligands: tuple[str, ...] | None = None
    wc_elements: tuple[str, ...] | None = None
    wc_cutoffs_ang: Mapping[tuple[str, str], float] | None = None


@dataclass(frozen=True)
class OrderOptions:
    """Inputs of the md_order analyses.

    ``neighbours``: 'bond valence' (the CN's bonds, the default, so a
    distortion index measures the polyhedron the CN counts), 'distance'
    (``distance_cutoffs_ang``, or the glass cation-anion cutoffs when None)
    or 'voronoi' (the cell faces). ``degrees`` and ``q_tet_selection`` are
    md_order's own defaults. The histogram bin widths (``*_bin``) are
    resolution choices; the edges run over the values measured, in
    multiples of the width. ``min_face_area_ang2`` and ``min_edge_ang`` are
    the Voronoi index thresholds (md_order's default, 0: no filtering).
    """

    neighbours: str = "bond valence"
    distance_cutoffs_ang: Mapping[tuple[str, str], float] | None = None
    degrees: tuple[int, ...] = md_order.DEGREES_DEFAULT
    q_tet_selection: str = "exactly four"
    q_bin: float = 0.01
    w_bin: float = 0.005
    q_tet_bin: float = 0.01
    baur_bin: float = 0.001
    angle_variance_bin_deg2: float = 1.0
    elongation_bin: float = 0.001
    volume_bin_ang3: float = 0.05
    ecn_bin: float = 0.05
    min_face_area_ang2: float = 0.0
    min_edge_ang: float = 0.0
    cell_volume_bin_ang3: float = 0.1


@dataclass(frozen=True)
class VoidOptions:
    """Inputs of the empty spheres and the free volume.

    ``radii`` is required: 'vdw' (FACET's van der Waals radii,
    ``md_order.vdw_radii_ang``, with ``md_order.VDW_RADII_SOURCE``), 'zero'
    (radii 0, the circumspheres; empty spheres only) or a map element -> Å
    with ``radii_source``. ``probe_radius_ang`` and ``grid_spacing_ang`` are
    required for the free volume (md_order requires them).
    ``sphere_bin_ang`` is the resolution of the empty-sphere histogram.
    """

    radii: str | Mapping[str, float] | None = None
    radii_source: str | None = None
    probe_radius_ang: float | None = None
    grid_spacing_ang: float | None = None
    sphere_bin_ang: float = 0.02


@dataclass(frozen=True)
class FractionMeasurement:
    """Measured fractions to put beside a glass descriptor (nmr-comparison).

    ``model_descriptor`` names a glass descriptor ('Qn Si',
    'N4 (B3, B4, other)', 'Al CN (4, 5, 6, other)', 'O speciation',
    'CN Al'); it is compared under both bond definitions when both exist.
    """

    model_descriptor: str
    measured: md_spectroscopy.MeasuredFractions


@dataclass(frozen=True)
class NmrOptions:
    """Inputs of the NMR shifts (md_spectroscopy).

    ``correlation`` is required, with its coefficients and reference (none
    ships); ``delta_ppm`` is the ppm axis (first, last, step), ``fwhm_ppm``
    and ``lineshape`` the broadening, all required. ``measured_fractions``
    lists measured sets for nmr-comparison.
    """

    correlation: md_spectroscopy.Correlation | None = None
    delta_ppm: tuple[float, float, float] | None = None
    fwhm_ppm: float | None = None
    lineshape: str | None = None
    measured_fractions: tuple[FractionMeasurement, ...] = ()


@dataclass(frozen=True)
class ExafsOptions:
    """Inputs of the EXAFS shells (md_spectroscopy).

    ``absorber`` is required. ``r_max_ang`` None takes the glass g(r) radius
    (the bond-valence search radius); ``dr_ang`` pdf.py's step. A neighbour's
    first-shell limit is ``first_shell_limits_ang[neighbour]`` or the first
    minimum of the frame-averaged absorber g(r) by the glass minimum rule.
    ``shells`` adds (neighbour, r_lo_ang, r_hi_ang) shells; ``weighting`` is
    md_spectroscopy's default 'none'.
    """

    absorber: str | None = None
    r_max_ang: float | None = None
    dr_ang: float = glass.RDF_DR_ANG
    neighbours: tuple[str, ...] | None = None
    first_shell_limits_ang: Mapping[str, float] | None = None
    shells: tuple[tuple[str, float, float], ...] = ()
    weighting: str = "none"


@dataclass(frozen=True)
class FeffOptions:
    """Inputs of the FEFF export (``md_spectroscopy.export_feff_inputs``),
    all required but ``r_path_ang`` and ``overwrite``; the absorber is the
    EXAFS one."""

    out_dir: str | None = None
    n_clusters: int | None = None
    seed: int | None = None
    cluster_radius_ang: float | None = None
    edge: str | None = None
    r_path_ang: float | None = None
    overwrite: bool = False


@dataclass(frozen=True)
class DynamicsOptions:
    """Inputs of the md_dynamics analyses.

    The time axis is the request's ``timestep_fs`` or ``frame_interval_ps``,
    or the file's own frame times. ``fit_t_min_ps`` / ``fit_t_max_ps`` (the
    MSD fit window) are required for the diffusion coefficient and the
    conductivities; ``lag_t_ps`` for the self and distinct correlations;
    ``van_hove_edges_r_ang`` (first, last, step), ``distinct_pairs`` and
    ``distinct_n_origins`` for the distinct van Hove function. The others are
    md_dynamics's own defaults. ``green_kubo_t_max_ps`` adds the Green-Kubo
    diffusion coefficient; ``residence_method`` the residence times.
    """

    unwrap: str = "auto"
    step_limit_fraction: float = md_dynamics.UNWRAP_STEP_LIMIT
    elements: tuple[str, ...] | None = None
    n_blocks: int = 1
    remove_com_drift: bool = False
    max_lag_t_ps: float | None = None
    fit_t_min_ps: float | None = None
    fit_t_max_ps: float | None = None
    lag_t_ps: tuple[float, ...] | None = None
    van_hove_edges_r_ang: tuple[float, float, float] | None = None
    isf_q_inv_ang: tuple[float, ...] | None = None
    distinct_pairs: tuple[tuple[str, str], ...] | None = None
    distinct_n_origins: int | None = None
    velocities: str = "file"
    vdos_window: str = "hann"
    green_kubo_t_max_ps: float | None = None
    lifetime_max_lag_t_ps: float | None = None
    gap_tolerance_frames: int = 0
    residence_method: str | None = None
    residence_t_min_ps: float | None = None
    residence_t_max_ps: float | None = None


@dataclass(frozen=True)
class AnalysisRequest:
    """Which analyses to run on a model, and every input they take.

    ``analyses``: names from :data:`ANALYSES`. ``formers``: the network
    formers (cation elements), :data:`NO_FORMERS` to state that none are
    named (the former-dependent descriptors are left out), or None when not
    given, which refuses every analysis that needs them. ``frames``: a
    slice of the readable frames (``slice(0, 100, 5)``) or their indices;
    None takes every one. ``v_bond_vu`` / ``v_list_vu``: FACET's thresholds
    (``bv.V_BOND_DEFAULT`` / ``bv.V_LIST_DEFAULT``). ``params``: the
    bond-valence parameter set (None: ``bv.DEFAULT``). ``ox_overrides``:
    oxidation states for ``md_model.model_oxidation``. ``bridging_anions``:
    None counts every anion of the model as a possible bridge (glass's
    default). ``type_map`` and ``read_options`` are what
    :func:`read_model` passes to ``md_readers.read_trajectory``. The
    physical inputs ``timestep_fs`` / ``frame_interval_ps`` (time axis),
    ``temperature_k`` and ``charges_e`` (a map element -> e, or one of
    :data:`CHARGE_SOURCES`) have no default. The option groups hold the rest.
    """

    analyses: tuple[str, ...]
    formers: frozenset[str] | str | None = None
    frames: slice | tuple[int, ...] | None = None
    v_bond_vu: float = bv.V_BOND_DEFAULT
    v_list_vu: float = bv.V_LIST_DEFAULT
    params: bv.ParameterSet | None = None
    ox_overrides: Mapping[str, int] = field(default_factory=dict)
    bridging_anions: tuple[str, ...] | None = None
    type_map: Mapping[int | str, str] | None = None
    read_options: Mapping[str, object] = field(default_factory=dict)
    timestep_fs: float | None = None
    frame_interval_ps: float | None = None
    temperature_k: float | None = None
    charges_e: Mapping[str, float] | str | None = None
    glass: GlassOptions = field(default_factory=GlassOptions)
    scattering: ScatteringOptions = field(default_factory=ScatteringOptions)
    network: NetworkOptions = field(default_factory=NetworkOptions)
    order: OrderOptions = field(default_factory=OrderOptions)
    voids: VoidOptions = field(default_factory=VoidOptions)
    nmr: NmrOptions = field(default_factory=NmrOptions)
    exafs: ExafsOptions = field(default_factory=ExafsOptions)
    feff: FeffOptions = field(default_factory=FeffOptions)
    dynamics: DynamicsOptions = field(default_factory=DynamicsOptions)

    def __post_init__(self) -> None:
        put = object.__setattr__
        if isinstance(self.analyses, str):
            raise ValueError("analyses needs a sequence of analysis names, "
                             "not one string")
        names = tuple(str(a) for a in self.analyses)
        unknown = [a for a in names if a not in ANALYSES]
        if unknown:
            raise ValueError(f"unknown analysis {', '.join(unknown)}; the "
                             f"analyses are {', '.join(ANALYSES)}")
        if len(set(names)) != len(names):
            raise ValueError("an analysis is named more than once")
        put(self, "analyses", tuple(a for a in ANALYSES if a in names))
        if self.formers is not None and self.formers != NO_FORMERS:
            if isinstance(self.formers, str):
                raise ValueError(f"formers needs a collection of element "
                                 f"symbols or {NO_FORMERS!r}, not "
                                 f"{self.formers!r}")
            put(self, "formers", frozenset(validate_symbol(str(s))
                                           for s in self.formers))
        if self.bridging_anions is not None:
            put(self, "bridging_anions", tuple(validate_symbol(str(s))
                                               for s in self.bridging_anions))
        put(self, "ox_overrides", dict(self.ox_overrides))
        put(self, "read_options", dict(self.read_options))
        for name in ("v_bond_vu", "v_list_vu"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} {value!r}: a number of 0 or more is "
                                 "needed")
        if self.charges_e is not None and isinstance(self.charges_e, str) \
                and self.charges_e not in CHARGE_SOURCES:
            raise ValueError(f"charges_e {self.charges_e!r}: a map element -> "
                             f"charge, or one of {CHARGE_SOURCES}")

    @property
    def named_formers(self) -> frozenset[str] | None:
        """The formers as a set; None when not given or stated as none."""
        if self.formers is None or self.formers == NO_FORMERS:
            return None
        return self.formers


# ---------------------------------------------------------------------------
# what a request lacks, before any frame is read
# ---------------------------------------------------------------------------

_TIME_AXIS = ("timestep_fs", "the MD timestep in fs (the file's timesteps "
              "times it give each frame's time), or frame_interval_ps, or a "
              "file that states its frame times; no timestep is assumed")


def _needs_formers(request: AnalysisRequest) -> list[MissingInput]:
    out = []
    net = request.network
    why_set = "the network formers (cation elements, e.g. Si,B)"
    if "glass" in request.analyses and request.formers is None:
        out.append(MissingInput(
            "glass", "formers", why_set + "; give 'none' to run glass without "
            "the former-dependent descriptors (speciation, Qn, connectivity)"))
    graph_users = [a for a in ("rings", "coordination-sequences", "components")
                   if a in request.analyses]
    for a in graph_users:
        needs = (net.graph == "bridging anion"
                 or (net.graph == "bond valence" and net.graph_elements is None)
                 or (a == "rings" and net.ring_t_elements is None)
                 or (a == "coordination-sequences" and net.cseq_centres is None))
        if needs and request.named_formers is None:
            out.append(MissingInput(
                a, "formers", why_set + f"; the '{net.graph}' graph and its "
                "defaults are built from them (or give the graph elements, "
                "centres and T elements explicitly)"))
    if "polyhedral-sharing" in request.analyses and \
            net.sharing_centres is None and request.named_formers is None:
        out.append(MissingInput("polyhedral-sharing", "formers", why_set
                                + "; they are the polyhedron centres unless "
                                "network.sharing_centres names them"))
    if "nmr" in request.analyses and request.nmr.correlation is not None:
        needs = {t.descriptor for t in request.nmr.correlation.terms} & {
            "T-O-T angle", "Qn", "bridges to"}
        if needs and request.named_formers is None:
            out.append(MissingInput("nmr", "formers", why_set + f"; the "
                                    f"correlation's {', '.join(sorted(needs))} "
                                    "term(s) are counted over them"))
    return out


def missing_inputs(request: AnalysisRequest) -> tuple[MissingInput, ...]:
    """Every input the requested analyses need and the request lacks, then
    every value the request gives that no analysis can take
    (:func:`invalid_inputs`).

    Checked without reading the model; :func:`analyse` checks the rest
    against the first frame (elements present, radii, charges, the time
    axis) and raises :class:`RequestError` with all of them.
    """
    a = set(request.analyses)
    out = _needs_formers(request)
    sc, net, order, voids = (request.scattering, request.network,
                             request.order, request.voids)
    nmr, exafs, feff, dyn = (request.nmr, request.exafs, request.feff,
                             request.dynamics)
    auto_glass = "the first minima of the glass analysis's g(r), so add glass "

    def who(*names: str) -> str:
        """The requested analyses among ``names``, joined with '/'."""
        return "/".join(n for n in names if n in a) or names[0]

    if "scattering" in a or "scattering-comparison" in a:
        who_sc = who("scattering", "scattering-comparison")
        if sc.r_window is None:
            out.append(MissingInput(who_sc, "scattering.r_window",
                                    "'none' or 'Lorch', the M(r) of the sine "
                                    "transform; it changes S(Q), and "
                                    "md_scattering states no default"))
        if not sc.radiations:
            out.append(MissingInput(who_sc, "scattering.radiations",
                                    "one or more of X-ray, neutron, electron"))
    if "scattering-comparison" in a:
        if "scattering" not in a:
            out.append(MissingInput("scattering-comparison", "analyses",
                                    "it compares the scattering analysis's "
                                    "curves: add scattering"))
        if not sc.measured:
            out.append(MissingInput("scattering-comparison",
                                    "scattering.measured",
                                    "a measured curve (file, radiation, "
                                    "function)"))
    if "rings" in a:
        if net.ring_criterion is None:
            out.append(MissingInput("rings", "network.ring_criterion",
                                    "king, guttman or primitive: the "
                                    "definition of a ring"))
        if net.ring_max_size is None:
            out.append(MissingInput("rings", "network.ring_max_size",
                                    "the largest ring searched, in graph "
                                    "nodes; the cost grows with it"))
    if "coordination-sequences" in a and net.n_shells is None:
        out.append(MissingInput("coordination-sequences", "network.n_shells",
                                "how many shells k to count"))
    if net.graph == "distance" and a & {"rings", "coordination-sequences",
                                        "components"}:
        who_graph = who("rings", "coordination-sequences", "components")
        if net.graph_elements is None:
            out.append(MissingInput(who_graph, "network.graph_elements",
                                    "the node elements of the distance graph"))
        if net.distance_cutoffs_ang is None and "glass" not in a:
            out.append(MissingInput(who_graph,
                                    "network.distance_cutoffs_ang",
                                    "the distance graph's cutoffs, or "
                                    + auto_glass + "to the analyses"))
    if "warren-cowley" in a and net.wc_cutoffs_ang is None and "glass" not in a:
        out.append(MissingInput("warren-cowley", "network.wc_cutoffs_ang",
                                "cutoffs for every element pair, or "
                                + auto_glass + "to the analyses"))
    if a & {"bond-order", "tetrahedral-order", "polyhedron-shape"} and \
            order.neighbours == "distance" and \
            order.distance_cutoffs_ang is None and "glass" not in a:
        out.append(MissingInput(who("bond-order", "tetrahedral-order",
                                    "polyhedron-shape"),
                                "order.distance_cutoffs_ang",
                                "the neighbour cutoffs, or " + auto_glass
                                + "to the analyses"))
    if a & {"empty-spheres", "free-volume"}:
        who_voids = who("empty-spheres", "free-volume")
        if voids.radii is None:
            out.append(MissingInput(who_voids, "voids.radii",
                                    "'vdw' (FACET's van der Waals radii), "
                                    "'zero', or element=radius pairs with "
                                    "voids.radii_source; no radius is "
                                    "assumed"))
        elif isinstance(voids.radii, Mapping) and not voids.radii_source:
            out.append(MissingInput(who_voids, "voids.radii_source",
                                    "where the radii given come from"))
    if "free-volume" in a:
        if voids.probe_radius_ang is None:
            out.append(MissingInput("free-volume", "voids.probe_radius_ang",
                                    "the probe radius in Å (0 gives the "
                                    "geometric fraction)"))
        if voids.grid_spacing_ang is None:
            out.append(MissingInput("free-volume", "voids.grid_spacing_ang",
                                    "the grid spacing in Å, the one "
                                    "approximation of the measurement"))
        if voids.radii == "zero":
            out.append(MissingInput("free-volume", "voids.radii",
                                    "radii of 0 leave no volume occupied; a "
                                    "radius set is needed"))
    if "nmr" in a or "nmr-comparison" in a:
        if "nmr" in a:
            if nmr.correlation is None:
                out.append(MissingInput("nmr", "nmr.correlation",
                                        "the published structure-shift "
                                        "correlation, its coefficients and "
                                        "reference (TODO: need reference; "
                                        "none ships)"))
            elif nmr.correlation.missing():
                out.append(MissingInput("nmr", "nmr.correlation", "; ".join(
                    nmr.correlation.missing())))
            for name, why in (("delta_ppm", "the ppm axis: first, last, step"),
                              ("fwhm_ppm", "the line width in ppm"),
                              ("lineshape", "gaussian or lorentzian")):
                if getattr(nmr, name) is None:
                    out.append(MissingInput("nmr", f"nmr.{name}", why))
        if "nmr-comparison" in a:
            if "glass" not in a:
                out.append(MissingInput("nmr-comparison", "analyses",
                                        "it compares the glass fractions: "
                                        "add glass"))
            if not nmr.measured_fractions:
                out.append(MissingInput("nmr-comparison",
                                        "nmr.measured_fractions",
                                        "the measured fractions and their "
                                        "source"))
    if ("exafs" in a or "feff" in a) and exafs.absorber is None:
        out.append(MissingInput(who("exafs", "feff"), "exafs.absorber",
                                "the absorbing element"))
    if "feff" in a:
        for name, why in (("out_dir", "the directory the inputs go to"),
                          ("n_clusters", "how many absorbers to draw"),
                          ("seed", "the seed of the draw"),
                          ("cluster_radius_ang", "the cluster radius in Å"),
                          ("edge", "the absorption edge (K, L3, ...)")):
            if getattr(feff, name) is None:
                out.append(MissingInput("feff", f"feff.{name}", why))
    if "conductivity" in a:
        if request.temperature_k is None:
            out.append(MissingInput("conductivity", "temperature_k",
                                    "the temperature of the run in K; it is "
                                    "a property of the run, not of the "
                                    "file"))
        if request.charges_e is None:
            out.append(MissingInput("conductivity", "charges_e",
                                    "element=charge pairs, 'oxidation "
                                    "states' or 'file': formal and partial "
                                    "charges give different conductivities"))
    if "conductivity" in a and (dyn.fit_t_min_ps is None
                                or dyn.fit_t_max_ps is None):
        out.append(MissingInput("conductivity", "dynamics.fit_t_min_ps/"
                                "fit_t_max_ps", "the fit window in ps of the "
                                "MSD and charge-displacement lines"))
    if a & {"self-correlations", "distinct-van-hove"} and dyn.lag_t_ps is None:
        out.append(MissingInput(who("self-correlations", "distinct-van-hove"),
                                "dynamics.lag_t_ps", "the lag times in ps"))
    if "distinct-van-hove" in a:
        for name, why in (("van_hove_edges_r_ang", "the r edges: first, last, "
                                                   "step in Å"),
                          ("distinct_pairs", "the element pairs, e.g. Na-Na"),
                          ("distinct_n_origins", "time origins per lag")):
            if getattr(dyn, name) is None:
                out.append(MissingInput("distinct-van-hove",
                                        f"dynamics.{name}", why))
    return tuple(out) + invalid_inputs(request)


# ---------------------------------------------------------------------------
# values no analysis can take, before any frame is read
# ---------------------------------------------------------------------------

_SC = "scattering/scattering-comparison"
_GRAPH = "rings/coordination-sequences/components"
_ORDER = "bond-order/tetrahedral-order/polyhedron-shape"
_VOIDS = "empty-spheres/free-volume"
_TIME = "/".join(_TIMED)


def _number_problem(value, *, above: float | None = None,
                    at_least: float | None = None,
                    whole: bool = False) -> str | None:
    """Why ``value`` is not a finite number above ``above`` (or of at least
    ``at_least``, or a whole number); None when it is one."""
    if isinstance(value, (bool, np.bool_)):
        return f"{value!r} is not a number"
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return f"{value!r} is not a number"
    if not math.isfinite(number):
        return f"{value!r} is not a finite number"
    if whole and not number.is_integer():
        return f"{value!r} is not a whole number"
    if above is not None and not number > above:
        return f"{value!r}: a number above {above:g} is needed"
    if at_least is not None and not number >= at_least:
        return f"{value!r}: a number of {at_least:g} or more is needed"
    return None


class _Checks:
    """Collects the MissingInput of every value a request gives that no
    analysis can take."""

    def __init__(self) -> None:
        self.out: list[MissingInput] = []

    def add(self, analysis: str, name: str, why: str) -> None:
        self.out.append(MissingInput(analysis, name, why))

    def number(self, analysis: str, name: str, value, **kw) -> bool:
        """True when ``value`` is None or passes; else records why."""
        if value is None:
            return True
        problem = _number_problem(value, **kw)
        if problem:
            self.add(analysis, name, problem)
        return problem is None

    def choice(self, analysis: str, name: str, value,
               allowed: Sequence[str]) -> None:
        if value is not None and value not in allowed:
            self.add(analysis, name, f"{value!r} is not one of "
                                     f"{', '.join(map(repr, allowed))}")

    def symbols(self, analysis: str, name: str, values) -> None:
        if values is None:
            return
        if isinstance(values, str):
            self.add(analysis, name, "a collection of element symbols is "
                                     "needed, not one string")
            return
        for value in values:
            self.symbol(analysis, name, value)

    def symbol(self, analysis: str, name: str, value) -> bool:
        try:
            validate_symbol(str(value))
        except ValueError:
            self.add(analysis, name, f"{value!r} is not an element symbol")
            return False
        return True

    def pair_map(self, analysis: str, name: str, mapping) -> None:
        """Element-pair keys, distances above 0 Å."""
        if mapping is None:
            return
        for key, value in dict(mapping).items():
            try:
                _pair_key(key)
            except ValueError:
                self.add(analysis, name, f"{key!r} is not an element pair "
                                         "such as 'Si-O'")
            label = "-".join(map(str, key)) if isinstance(key, tuple) else key
            self.number(analysis, f"{name} {label}", value, above=0.0)

    def pair_of(self, analysis: str, name: str, value) -> None:
        """A (low, high) window given as one value."""
        if value is None:
            return
        values = tuple(value) if not isinstance(value, str) else (value,)
        if len(values) != 2:
            self.add(analysis, name, f"{value!r}: two numbers (low, high) "
                                     "are needed")
            return
        self.window(analysis, name, *values)

    def window(self, analysis: str, name: str, low, high, *,
               low_at_least: float = 0.0) -> None:
        """Two ends, the lower one >= low_at_least, the upper above it."""
        ok = self.number(analysis, name, low, at_least=low_at_least)
        ok = self.number(analysis, name, high, above=low_at_least) and ok
        if ok and low is not None and high is not None \
                and not float(high) > float(low):
            self.add(analysis, name, f"the window ({low!r}, {high!r}) does "
                                     "not run upwards")

    def triple(self, analysis: str, name: str, value, *,
               first_at_least: float | None = None) -> None:
        """(first, last, step): a step above 0 and last above first."""
        if value is None:
            return
        values = tuple(value)
        if len(values) != 3:
            self.add(analysis, name, "three numbers are needed: first, last, "
                                     "step")
            return
        first, last, step = values
        ok = self.number(analysis, name, first, at_least=first_at_least)
        ok = self.number(analysis, name, last) and ok
        ok = self.number(analysis, f"{name} step", step, above=0.0) and ok
        if ok and not float(last) > float(first):
            self.add(analysis, name, f"last {last!r} is not above first "
                                     f"{first!r}")


def invalid_inputs(request: AnalysisRequest) -> tuple[MissingInput, ...]:
    """Every value the request gives that no analysis can take.

    Every option given is checked, whichever analyses the request names, so
    an impossible value is refused wherever it stands: a step or bin width
    that is not a finite number above 0, a (first, last, step) triple or a
    window that does not run upwards, a name outside the module's own list
    (the ring criteria of md_network, the radiations, windows and FSDP
    baselines of md_scattering, the lineshapes and weightings of
    md_spectroscopy, the unwrap, velocity, window and residence methods of
    md_dynamics), a text that is not an element symbol, a measured file
    that does not exist, a count below its least value, and the
    contradictions: both timestep_fs and frame_interval_ps, a frame step of
    0, v_bond below v_list. The model is not read (module docstring). Each
    item names the analyses that read the option, or 'request' for a field
    read by many.
    """
    c = _Checks()
    req = request
    # -- the request's own fields ---------------------------------------------
    if req.v_bond_vu < req.v_list_vu:
        c.add("request", "v_bond_vu", (
            f"v_bond_vu {req.v_bond_vu!r} v.u. lies below v_list_vu "
            f"{req.v_list_vu!r} v.u.: a contact below v_list is never "
            "tabulated, so every CN, bond and Qn would be the one at v_list "
            "while labelled v_bond; give v_list_vu at or below v_bond_vu"))
    if isinstance(req.frames, slice):
        for part in ("start", "stop", "step"):
            value = getattr(req.frames, part)
            if value is not None and (isinstance(value, (bool, np.bool_))
                                      or not isinstance(value, (int,
                                                                np.integer))):
                c.add("request", "frames", f"the {part} {value!r} is not a "
                                           "whole number")
        if req.frames.step == 0:
            c.add("request", "frames", "a step of 0 selects no frame")
    elif req.frames is not None:
        for k in req.frames:
            if isinstance(k, (bool, np.bool_)) or \
                    not isinstance(k, (int, np.integer)) or k < 0:
                c.add("request", "frames", f"frame index {k!r} is not a whole "
                                           "number of 0 or more")
    if req.timestep_fs is not None and req.frame_interval_ps is not None:
        c.add(_TIME, "timestep_fs", "timestep_fs and frame_interval_ps are "
              "both given: they are two statements of one time axis; give "
              "one")
    c.number(_TIME, "timestep_fs", req.timestep_fs, above=0.0)
    c.number(_TIME, "frame_interval_ps", req.frame_interval_ps, above=0.0)
    c.number("conductivity", "temperature_k", req.temperature_k, above=0.0)
    if isinstance(req.charges_e, Mapping):
        for key, value in req.charges_e.items():
            c.symbol("conductivity", "charges_e", key)
            c.number("conductivity", f"charges_e {key}", value)
    for key, value in dict(req.ox_overrides).items():
        c.symbol("request", "ox_overrides", key)
        c.number("request", f"ox_overrides {key}", value, whole=True)

    # -- glass ----------------------------------------------------------------
    g = req.glass
    c.number("glass", "glass.rdf_r_max_ang", g.rdf_r_max_ang, above=0.0)
    c.number("glass", "glass.rdf_dr_ang", g.rdf_dr_ang, above=0.0)
    for build, name in ((g.minimum, "glass.minimum_*"),
                        (g.bins, "glass bin widths")):
        try:
            build()
        except (ValueError, TypeError) as error:
            c.add("glass", name, str(error))
    c.pair_map("glass", "glass.cutoffs_ang", g.cutoffs_ang)
    if g.oxide_basis is not None:
        for key in dict(g.oxide_basis):
            c.symbol("glass", "glass.oxide_basis", key)

    # -- scattering -----------------------------------------------------------
    s = req.scattering
    c.choice(_SC, "scattering.r_window", s.r_window, md_scattering.R_WINDOWS)
    if s.radiations is not None:
        for radiation in s.radiations:
            c.choice(_SC, "scattering.radiations", radiation,
                     md_scattering.RADIATIONS)
    c.number(_SC, "scattering.r_max_ang", s.r_max_ang, above=0.0)
    if c.number(_SC, "scattering.dr_ang", s.dr_ang, above=0.0) \
            and s.r_max_ang is not None and \
            _number_problem(s.r_max_ang, above=0.0) is None and \
            not float(s.dr_ang) < float(s.r_max_ang):
        c.add(_SC, "scattering.dr_ang", f"dr_ang {s.dr_ang!r} is not below "
                                        f"r_max_ang {s.r_max_ang!r}")
    if c.number(_SC, "scattering.q_step_inv_ang", s.q_step_inv_ang,
                above=0.0) and c.number(_SC, "scattering.q_grid_max_inv_ang",
                                        s.q_grid_max_inv_ang, above=0.0) \
            and not float(s.q_grid_max_inv_ang) > float(s.q_step_inv_ang):
        c.add(_SC, "scattering.q_grid_max_inv_ang",
              f"{s.q_grid_max_inv_ang!r} is not above q_step_inv_ang "
              f"{s.q_step_inv_ang!r}")
    q_max, q_min = s.termination_q_max_inv_ang, s.termination_q_min_inv_ang
    c.number(_SC, "scattering.termination_q_max_inv_ang", q_max, above=0.0)
    c.number(_SC, "scattering.termination_q_min_inv_ang", q_min, at_least=0.0)
    if q_max is None:
        if q_min is not None:
            c.add(_SC, "scattering.termination_q_min_inv_ang",
                  "given without termination_q_max_inv_ang; a termination "
                  "range needs its upper end (and q_window)")
        if s.q_window is not None:
            c.add(_SC, "scattering.q_window", "given without "
                  "termination_q_max_inv_ang")
    else:
        if s.q_window is None:
            c.add(_SC, "scattering.q_window", f"needed with "
                  f"termination_q_max_inv_ang: one of "
                  f"{', '.join(md_scattering.Q_WINDOWS)} (no default; the "
                  "choice changes G(r))")
        if q_min is not None and _number_problem(q_max, above=0.0) is None \
                and _number_problem(q_min, at_least=0.0) is None \
                and not float(q_min) < float(q_max):
            c.add(_SC, "scattering.termination_q_min_inv_ang",
                  f"{q_min!r} is not below termination_q_max_inv_ang "
                  f"{q_max!r}")
    c.choice(_SC, "scattering.q_window", s.q_window, md_scattering.Q_WINDOWS)
    if s.fsdp_window_inv_ang is not None:
        c.pair_of(_SC, "scattering.fsdp_window_inv_ang", s.fsdp_window_inv_ang)
        if s.fsdp_baseline is None:
            c.add(_SC, "scattering.fsdp_baseline", f"needed with "
                  f"fsdp_window_inv_ang: one of "
                  f"{', '.join(md_scattering.FSDP_BASELINES)}")
    elif s.fsdp_baseline is not None:
        c.add(_SC, "scattering.fsdp_baseline", "given without "
                                               "fsdp_window_inv_ang")
    c.choice(_SC, "scattering.fsdp_baseline", s.fsdp_baseline,
             md_scattering.FSDP_BASELINES)
    if s.neutron_lengths_fm is not None:
        try:
            md_scattering._neutron_overrides(
                s.neutron_lengths_fm, s.lengths_source,
                list(s.neutron_lengths_fm))
        except ValueError as error:
            c.add(_SC, "scattering.neutron_lengths_fm", str(error))
    for n, spec in enumerate(s.measured):
        where = f"scattering.measured[{n}]"
        if not isinstance(spec, MeasuredCurve):
            c.add("scattering-comparison", where, "a MeasuredCurve is needed")
            continue
        c.choice("scattering-comparison", f"{where}.function", spec.function,
                 tuple(CURVE_FUNCTIONS))
        c.choice("scattering-comparison", f"{where}.radiation",
                 spec.radiation, md_scattering.RADIATIONS)
        if s.radiations and spec.radiation in md_scattering.RADIATIONS and \
                spec.radiation not in s.radiations:
            c.add("scattering-comparison", f"{where}.radiation",
                  f"{spec.radiation} is not among scattering.radiations "
                  f"({', '.join(s.radiations)}), so no model curve is "
                  "computed for it")
        if spec.function == "G(r) from S(Q)" and (q_max is None
                                                  or s.q_window is None):
            c.add("scattering-comparison", f"{where}.function",
                  "G(r) from S(Q) needs scattering.termination_q_max_inv_ang "
                  "and scattering.q_window")
        if not Path(str(spec.path)).is_file():
            c.add("scattering-comparison", f"{where}.path",
                  f"{spec.path}: no such file")
        c.pair_of("scattering-comparison", f"{where}.axis_range",
                  spec.axis_range)
        c.number("scattering-comparison", f"{where}.scale", spec.scale)

    # -- network --------------------------------------------------------------
    net = req.network
    c.choice(_GRAPH, "network.graph", net.graph, GRAPH_KINDS)
    c.choice("rings", "network.ring_criterion", net.ring_criterion,
             md_network.CRITERIA)
    c.number("rings", "network.ring_max_size", net.ring_max_size, at_least=3,
             whole=True)
    c.number("coordination-sequences", "network.n_shells", net.n_shells,
             at_least=1, whole=True)
    for name, analysis in (("graph_elements", _GRAPH),
                           ("ring_t_elements", "rings"),
                           ("cseq_centres", "coordination-sequences"),
                           ("sharing_centres", "polyhedral-sharing"),
                           ("sharing_ligands", "polyhedral-sharing"),
                           ("wc_elements", "warren-cowley")):
        c.symbols(analysis, f"network.{name}", getattr(net, name))
    c.pair_map(_GRAPH, "network.distance_cutoffs_ang",
               net.distance_cutoffs_ang)
    c.pair_map("warren-cowley", "network.wc_cutoffs_ang", net.wc_cutoffs_ang)

    # -- order ----------------------------------------------------------------
    o = req.order
    c.choice(_ORDER, "order.neighbours", o.neighbours, NEIGHBOUR_KINDS)
    c.choice("tetrahedral-order", "order.q_tet_selection", o.q_tet_selection,
             md_order.QTET_SELECTIONS)
    degrees = tuple(o.degrees) if not isinstance(o.degrees, (int, str)) \
        else (o.degrees,)
    if not degrees:
        c.add("bond-order", "order.degrees", "no degree l was given")
    if len(set(degrees)) != len(degrees):
        c.add("bond-order", "order.degrees", f"a degree occurs twice in "
                                             f"{degrees}")
    for degree in degrees:
        problem = _number_problem(degree, at_least=1, whole=True)
        if problem is None and int(degree) > md_order.DEGREE_MAX:
            problem = (f"l = {degree}: at most {md_order.DEGREE_MAX} "
                       "(md_order.DEGREE_MAX, a cost bound)")
        if problem:
            c.add("bond-order", "order.degrees", problem)
    for name in ("q_bin", "w_bin", "q_tet_bin", "baur_bin",
                 "angle_variance_bin_deg2", "elongation_bin",
                 "volume_bin_ang3", "ecn_bin"):
        c.number(_ORDER, f"order.{name}", getattr(o, name), above=0.0)
    c.number("voronoi", "order.cell_volume_bin_ang3", o.cell_volume_bin_ang3,
             above=0.0)
    for name in ("min_face_area_ang2", "min_edge_ang"):
        c.number("voronoi/" + _ORDER, f"order.{name}", getattr(o, name),
                 at_least=0.0)
    c.pair_map(_ORDER, "order.distance_cutoffs_ang", o.distance_cutoffs_ang)

    # -- voids ----------------------------------------------------------------
    v = req.voids
    if isinstance(v.radii, Mapping):
        for key, value in v.radii.items():
            c.symbol(_VOIDS, "voids.radii", key)
            c.number(_VOIDS, f"voids.radii {key}", value, at_least=0.0)
    elif v.radii is not None:
        c.choice(_VOIDS, "voids.radii", v.radii, RADII_NAMES)
    c.number("free-volume", "voids.probe_radius_ang", v.probe_radius_ang,
             at_least=0.0)
    c.number("free-volume", "voids.grid_spacing_ang", v.grid_spacing_ang,
             above=0.0)
    c.number("empty-spheres", "voids.sphere_bin_ang", v.sphere_bin_ang,
             above=0.0)

    # -- NMR and EXAFS --------------------------------------------------------
    nmr = req.nmr
    c.triple("nmr", "nmr.delta_ppm", nmr.delta_ppm)
    c.number("nmr", "nmr.fwhm_ppm", nmr.fwhm_ppm, above=0.0)
    c.choice("nmr", "nmr.lineshape", nmr.lineshape,
             md_spectroscopy.LINESHAPES)
    x = req.exafs
    if x.absorber is not None:
        c.symbol("exafs/feff", "exafs.absorber", x.absorber)
    c.number("exafs", "exafs.r_max_ang", x.r_max_ang, above=0.0)
    c.number("exafs", "exafs.dr_ang", x.dr_ang, above=0.0)
    c.symbols("exafs", "exafs.neighbours", x.neighbours)
    for key, value in dict(x.first_shell_limits_ang or {}).items():
        c.symbol("exafs", "exafs.first_shell_limits_ang", key)
        c.number("exafs", f"exafs.first_shell_limits_ang {key}", value,
                 above=0.0)
    for shell in x.shells:
        if len(tuple(shell)) != 3:
            c.add("exafs", "exafs.shells", f"{shell!r}: (element, r_lo_ang, "
                                           "r_hi_ang) is needed")
            continue
        element, low, high = shell
        c.symbol("exafs", "exafs.shells", element)
        c.window("exafs", f"exafs.shells {element}", low, high)
    c.choice("exafs", "exafs.weighting", x.weighting,
             md_spectroscopy.WEIGHTINGS)
    f = req.feff
    if None not in (f.out_dir, f.n_clusters, f.seed, f.cluster_radius_ang,
                    f.edge):
        try:
            md_spectroscopy.FeffRequest(
                f.out_dir, f.n_clusters, f.seed, f.cluster_radius_ang, f.edge,
                r_path_ang=f.r_path_ang, overwrite=f.overwrite)
        except (ValueError, TypeError) as error:
            c.add("feff", "feff", str(error))

    # -- dynamics -------------------------------------------------------------
    d = req.dynamics
    c.choice(_TIME, "dynamics.unwrap", d.unwrap, md_dynamics.UNWRAP_METHODS)
    if c.number(_TIME, "dynamics.step_limit_fraction", d.step_limit_fraction,
                above=0.0) and d.unwrap == "minimum image" \
            and float(d.step_limit_fraction) > 0.5:
        c.add(_TIME, "dynamics.step_limit_fraction",
              f"{d.step_limit_fraction!r}: above 0.5 a step and its periodic "
              "image cannot be told apart by continuity")
    c.symbols(_TIME, "dynamics.elements", d.elements)
    c.number(_TIME, "dynamics.n_blocks", d.n_blocks, at_least=1, whole=True)
    c.number(_TIME, "dynamics.max_lag_t_ps", d.max_lag_t_ps, above=0.0)
    if (d.fit_t_min_ps is None) != (d.fit_t_max_ps is None):
        c.add("msd/conductivity", "dynamics.fit_t_min_ps/fit_t_max_ps",
              "one end of the MSD fit window is given; the fit needs both")
    elif d.fit_t_min_ps is not None:
        c.window("msd/conductivity", "dynamics.fit_t_min_ps/fit_t_max_ps",
                 d.fit_t_min_ps, d.fit_t_max_ps)
    if d.lag_t_ps is not None:
        if not tuple(d.lag_t_ps):
            c.add("self-correlations/distinct-van-hove", "dynamics.lag_t_ps",
                  "no lag time was given")
        for lag in d.lag_t_ps:
            c.number("self-correlations/distinct-van-hove",
                     "dynamics.lag_t_ps", lag, at_least=0.0)
    c.triple("self-correlations/distinct-van-hove",
             "dynamics.van_hove_edges_r_ang", d.van_hove_edges_r_ang,
             first_at_least=0.0)
    for q in d.isf_q_inv_ang or ():
        c.number("self-correlations", "dynamics.isf_q_inv_ang", q, above=0.0)
    for pair in d.distinct_pairs or ():
        try:
            _pair_key(pair)
        except ValueError:
            c.add("distinct-van-hove", "dynamics.distinct_pairs",
                  f"{pair!r} is not an element pair such as 'Na-Na'")
    c.number("distinct-van-hove", "dynamics.distinct_n_origins",
             d.distinct_n_origins, at_least=1, whole=True)
    c.choice("vacf/kinetic-temperature", "dynamics.velocities", d.velocities,
             md_dynamics.VELOCITY_SOURCES)
    c.choice("vacf", "dynamics.vdos_window", d.vdos_window,
             md_dynamics.WINDOWS)
    c.number("vacf", "dynamics.green_kubo_t_max_ps", d.green_kubo_t_max_ps,
             above=0.0)
    c.number("bond-lifetimes", "dynamics.lifetime_max_lag_t_ps",
             d.lifetime_max_lag_t_ps, above=0.0)
    c.number("bond-lifetimes", "dynamics.gap_tolerance_frames",
             d.gap_tolerance_frames, at_least=0, whole=True)
    c.choice("bond-lifetimes", "dynamics.residence_method",
             d.residence_method, md_dynamics.RESIDENCE_METHODS)
    if d.residence_method == "integral" and d.residence_t_min_ps is not None:
        c.add("bond-lifetimes", "dynamics.residence_t_min_ps",
              "the 'integral' residence time runs from 0; a t_min is given")
    if d.residence_method == "exponential fit":
        if d.residence_t_min_ps is None or d.residence_t_max_ps is None:
            c.add("bond-lifetimes", "dynamics.residence_t_min_ps/"
                  "residence_t_max_ps", "the 'exponential fit' needs both "
                  "ends of its window in ps")
        else:
            c.window("bond-lifetimes", "dynamics.residence_t_min_ps/"
                     "residence_t_max_ps", d.residence_t_min_ps,
                     d.residence_t_max_ps)
    else:
        c.number("bond-lifetimes", "dynamics.residence_t_max_ps",
                 d.residence_t_max_ps, above=0.0)
    return tuple(c.out)


def dependencies(request: AnalysisRequest, name: str) -> tuple[str, ...]:
    """The analyses ``name`` reads the results of, under this request: the
    comparisons read what they compare, and a distance definition without
    given cutoffs reads the glass first minima."""
    net, order = request.network, request.order
    if name == "nmr-comparison":
        return ("glass",)
    if name == "scattering-comparison":
        return ("scattering",)
    if name == "warren-cowley" and net.wc_cutoffs_ang is None:
        return ("glass",)
    if name in ("rings", "coordination-sequences", "components") and \
            net.graph == "distance" and net.distance_cutoffs_ang is None:
        return ("glass",)
    if name in ("bond-order", "tetrahedral-order", "polyhedron-shape") and \
            order.neighbours == "distance" and \
            order.distance_cutoffs_ang is None:
        return ("glass",)
    return ()


def available_analyses(request: AnalysisRequest, trajectory=None
                       ) -> tuple[tuple[str, ...], dict[str, list[str]]]:
    """(runnable, left out): the analyses whose inputs the request gives,
    and for each other analysis the inputs it lacks.

    Without ``trajectory`` the model is not read. With it, the chosen frames
    are checked against it (:class:`RequestError` when they select nothing)
    and so is what the file itself has to provide: an analysis of time
    needs a time axis (timestep_fs, frame_interval_ps, or frame times in the
    file) and two frames, and the VACF on the file's velocities and the
    kinetic temperature need velocities in the file. A dump without them
    leaves those analyses out, named with what they lack, rather than
    running them to a failure at the end. Raises :class:`FramesUnreadable`
    when no chosen frame can be read.
    """
    runnable, left = [], {}
    for name in ANALYSES:
        needed = dependencies(request, name)
        trial = dataclasses.replace(request, analyses=(name,) + needed)
        missing = [m for m in missing_inputs(trial)
                   if name in m.analysis.split("/")]
        lacking_deps = [d for d in needed if d in left]
        if missing:
            left[name] = sorted({m.name for m in missing})
        elif lacking_deps:
            left[name] = [f"{d} (it reads its results)" for d in lacking_deps]
        else:
            runnable.append(name)
    if trajectory is not None and any(n in _TIMED for n in runnable):
        frames = _chosen_frames(trajectory, request.frames)
        first, _ = _first_readable(trajectory, frames)
        for name in [n for n in runnable if n in _TIMED]:
            problems = _trajectory_problems(trajectory, request, frames, first,
                                            (name,))
            if problems:
                runnable.remove(name)
                left[name] = sorted({m.name for m in problems})
    return tuple(runnable), left


def _first_readable(trajectory: Trajectory, frames: Sequence[int]
                    ) -> tuple[Frame, dict[int, str]]:
    """The first chosen frame that can be read, and the reasons of those
    before it that could not."""
    unreadable: dict[int, str] = {}
    for k in frames:
        try:
            return trajectory.frame(k), unreadable
        except FrameError as error:
            unreadable[k] = f"could not be read: {error}"
    raise FramesUnreadable("no chosen frame could be read: " + "; ".join(
        f"frame {k}: {r}" for k, r in sorted(unreadable.items())))


def _trajectory_problems(trajectory: Trajectory, request: AnalysisRequest,
                         frames: Sequence[int], first: Frame,
                         analyses: Iterable[str]) -> list[MissingInput]:
    """What the file has to provide for the analyses of time among
    ``analyses``: two frames, a time axis (``md_dynamics.time_axis_ps``
    itself decides), and the velocities the VACF and the kinetic
    temperature read."""
    out = []
    wanted = set(analyses)
    timed = [a for a in _TIMED if a in wanted]
    if not timed:
        return out
    who = "/".join(timed)
    if len(frames) < 2:
        out.append(MissingInput(who, "frames", f"{len(frames)} frame chosen; "
                                "a function of time needs at least two"))
    else:
        try:
            md_dynamics.time_axis_ps(trajectory, list(frames),
                                     timestep_fs=request.timestep_fs,
                                     frame_interval_ps=request.frame_interval_ps)
        except ValueError as error:
            name = ("timestep_fs or frame_interval_ps"
                    if request.timestep_fs is None
                    and request.frame_interval_ps is None
                    else "timestep_fs" if request.timestep_fs is not None
                    else "frame_interval_ps")
            out.append(MissingInput(who, name, str(error)))
    needs_velocities = [a for a in timed if a == "kinetic-temperature" or (
        a == "vacf" and request.dynamics.velocities == "file")]
    if needs_velocities and first.vel_ang_per_ps is None:
        out.append(MissingInput(
            "/".join(needs_velocities), "velocities in the file",
            "the file holds no velocities (none in the first chosen frame); "
            "the kinetic temperature reads them, and the VACF reads them "
            "unless dynamics.velocities = 'finite difference' derives them "
            "from the positions"))
    return out


def read_model(path, request: AnalysisRequest | None = None, **options
               ) -> Trajectory:
    """``md_readers.read_trajectory`` with the request's type map and read
    options (and ``options``, which override them)."""
    from . import md_readers

    given = {}
    if request is not None:
        given.update(request.read_options)
        if request.type_map is not None:
            given["type_map"] = dict(request.type_map)
    given.update(options)
    return md_readers.read_trajectory(path, **given)


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Table:
    """Rows of one descriptor that is not an md_stats container (a list of
    minima, a composition, a comparison). Column names carry units.
    ``frames``: the frames the rows cover, when they are not every frame
    of the analysis (the EXAFS shells after a pass 2 that did not reach
    every frame); None when they are."""

    name: str
    rows: tuple[dict, ...]
    notes: tuple[str, ...] = ()
    frames: tuple[int, ...] | None = None

    def as_rows(self) -> list[dict]:
        return [dict(row) for row in self.rows]


@dataclass(frozen=True, eq=False)
class AnalysisOutput:
    """One analysis: its descriptors, its provenance, what it left out.

    ``tables`` maps a descriptor name to an md_stats container
    (Distribution, Histogram, Series, Scalar), a :class:`Table`, or a module
    result with ``as_rows()``. ``raw`` is the module's own result object
    (``GlassResult``, ``ScatteringResult``, the per-frame results...), for a
    caller that wants more than the tables. ``error`` is the reason when the
    analysis produced nothing.
    """

    name: str
    tables: Mapping[str, object]
    provenance: Provenance | None
    raw: object = None
    notes: tuple[str, ...] = ()
    error: str | None = None
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True, eq=False)
class ModelResult:
    """Everything one :func:`analyse` run measured.

    ``outputs`` maps each requested analysis to its :class:`AnalysisOutput`
    (in :data:`ANALYSES` order). ``provenance`` is the run's header: the
    file, the frames, the type map and its source, the oxidation states, the
    bond-valence set and thresholds, the formers, the distance cutoffs and
    every method parameter of the request. ``searches`` maps each frame to
    the ``bulk.iter_pairs`` calls this run made on it, per pass.
    ``minima`` holds the glass first minima (``RdfMinimum`` per ordered
    element pair) when glass ran. ``timings_s`` the wall time of each stage.
    ``model_notes``: what the first frame showed about the inputs, whatever
    the analyses (a composition that is not neutral with the oxidation
    states used, the per-atom charges of the file, named formers absent
    from the model); md_export writes them into every header and the
    command line prints them.
    """

    request: AnalysisRequest
    provenance: Provenance
    outputs: Mapping[str, AnalysisOutput]
    searches: Mapping[int, tuple[int, int]]
    minima: Mapping
    notes: tuple[str, ...]
    cancelled: bool
    timings_s: Mapping[str, float]
    model_notes: tuple[str, ...] = ()

    def descriptors(self) -> Iterator[tuple[str, str, object]]:
        """(analysis, descriptor name, container) of every table."""
        for name, out in self.outputs.items():
            for key, table in out.tables.items():
                yield name, key, table

    @property
    def failed(self) -> dict[str, str]:
        return {name: out.error for name, out in self.outputs.items()
                if out.error is not None}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _frozen(array) -> np.ndarray:
    out = np.array(array, copy=True)
    out.setflags(write=False)
    return out


def _restrict(block: bulk.PairTable, r_ang: float, vectors: bool
              ) -> bulk.PairTable:
    """The pairs of ``block`` with d <= r_ang: what a search to r_ang finds.

    The pairs are those of the wider search with d <= r_ang, with the same
    distances, vectors and images (formed from the fractions by one
    expression whatever the radius), so the copy is the block a search of
    the same frame to r_ang returns. ``vectors`` False keeps none
    (``vectors_within_ang`` 0), as a distance-only search.
    """
    if r_ang > block.r_ang:
        raise ValueError(f"a copy to {r_ang!r} Å was asked of pairs searched "
                         f"to {block.r_ang!r} Å")
    if r_ang == block.r_ang and not vectors:
        # every pair is kept: the block's own (read-only) arrays are shared,
        # not copied (0.2 s a frame on 4.7 million pairs, measured)
        return dataclasses.replace(block, vec_ang=None, vectors_within_ang=0.0)
    keep = block.d_ang <= r_ang
    vec = None
    within = 0.0
    if vectors:
        if block.vec_ang is None:
            raise ValueError("the shared search kept no vectors")
        vec = block.vec_ang[keep]
        if not np.isfinite(vec).all():
            raise ValueError(f"the shared search kept vectors only to "
                             f"{block.vectors_within_ang!r} Å, below "
                             f"{r_ang!r} Å")
        within = None
    return bulk.PairTable(
        i=_frozen(block.i[keep]), j=_frozen(block.j[keep]),
        image=_frozen(block.image[keep]), d_ang=_frozen(block.d_ang[keep]),
        vec_ang=None if vec is None else _frozen(vec), r_ang=float(r_ang),
        d_min_ang=block.d_min_ang, n_below_d_min=block.n_below_d_min,
        method=block.method, n_atoms=block.n_atoms,
        centre_start=block.centre_start, centre_stop=block.centre_stop,
        vectors_within_ang=within,
        closest_below_d_min=block.closest_below_d_min,
        frame_digest=block.frame_digest)


class _Sink:
    """Collects one analysis's restricted copy of a frame's pair blocks."""

    def __init__(self, name: str, r_ang: float, vectors: bool):
        self.name = name
        self.r_ang = float(r_ang)
        self.vectors = bool(vectors)
        self.blocks: list[bulk.PairTable] = []
        self.key: tuple | None = None
        self.complete = False

    def start(self, key: tuple) -> None:
        self.blocks = []
        self.key = key
        self.complete = False

    def add(self, block: bulk.PairTable) -> None:
        self.blocks.append(_restrict(block, self.r_ang, self.vectors))


def _pair_key(text) -> tuple[str, str]:
    if isinstance(text, tuple) and len(text) == 2:
        return validate_symbol(str(text[0])), validate_symbol(str(text[1]))
    parts = str(text).replace(" ", "").split("-")
    if len(parts) != 2:
        raise ValueError(f"{text!r} is not an element pair such as 'Si-O'")
    return validate_symbol(parts[0]), validate_symbol(parts[1])


def _pair_map(mapping) -> dict[tuple[str, str], float]:
    out = {}
    for key, value in dict(mapping).items():
        out[_pair_key(key)] = float(value)
    return out


def _symbols_of(values) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError("a collection of element symbols is needed")
    return tuple(validate_symbol(str(v)) for v in values)


def _chosen_frames(trajectory: Trajectory, frames) -> list[int]:
    """The readable frame indices ``frames`` selects, ascending;
    :class:`RequestError` when the selection does not fit the trajectory
    (an input contradicting the model, before any frame is read)."""

    def refuse(why: str):
        return RequestError([MissingInput("request", "frames", why)])

    if frames is None or isinstance(frames, slice):
        every = range(trajectory.n_frames)
        try:
            out = list(every if frames is None else every[frames])
        except (TypeError, ValueError) as error:
            raise refuse(f"{frames!r}: {error}") from None
        if not out:
            raise refuse(f"{_frames_text(frames)} selects no frame of the "
                         f"{trajectory.n_frames} readable ones (0 .. "
                         f"{trajectory.n_frames - 1})")
        return sorted(out)
    out = []
    for k in frames:
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int,
                                                                 np.integer)):
            raise refuse(f"frame index {k!r} is not an integer")
        if not 0 <= int(k) < trajectory.n_frames:
            raise refuse(f"frame {k} is outside the {trajectory.n_frames} "
                         f"readable frames (0 .. {trajectory.n_frames - 1})")
        out.append(int(k))
    if len(set(out)) != len(out):
        raise refuse("frames names a frame more than once")
    if not out:
        raise refuse("frames selects no frame")
    return sorted(out)


def _frames_text(frames: slice) -> str:
    """'50:60' for slice(50, 60, None), as the command line writes it."""
    parts = ["" if v is None else str(v)
             for v in (frames.start, frames.stop, frames.step)]
    if frames.step is None:
        parts = parts[:2]
    return "frames " + ":".join(parts)


def _edges_over(rows: Sequence[np.ndarray], width: float
                ) -> tuple[np.ndarray, tuple[str, ...]]:
    """Bin edges in multiples of ``width`` over the finite values, and notes.

    The edges span every finite value when that takes at most
    :data:`HISTOGRAM_BINS_MAX` bins. A few extreme values would otherwise
    stretch them: on a 2 880-atom aluminosilicate glass a polyhedron of
    nearly zero volume gave a quadratic elongation near 830, and edges of
    0.001 over all values made 830 000 bins per element (measured
    2026-10-07). Then the edges run between the
    :data:`HISTOGRAM_WINDOW_PERCENTILES` of the values pooled over the
    frames, at most HISTOGRAM_BINS_MAX bins from the lower one, and the
    values outside are counted in the histogram's ``n_below`` / ``n_above``
    (md_stats never drops them), with a note giving the full range.
    """
    if _number_problem(width, above=0.0) is not None:
        raise ValueError(f"bin width {width!r}: a finite number above 0 is "
                         "needed")
    finite = [r[np.isfinite(r)] for r in rows]
    values = [f for f in finite if f.size]
    if not values:
        return np.array([0.0, width]), ()
    low = min(float(v.min()) for v in values)
    high = max(float(v.max()) for v in values)
    notes: tuple[str, ...] = ()
    if (high - low) / width > HISTOGRAM_BINS_MAX:
        pooled = np.concatenate(values)
        lo_q, hi_q = (float(x) for x in np.percentile(
            pooled, HISTOGRAM_WINDOW_PERCENTILES))
        span_low = lo_q
        span_high = min(hi_q, lo_q + (HISTOGRAM_BINS_MAX - 1) * width)
        notes = (f"the values run from {low!r} to {high!r}, which bins of "
                 f"{width:g} would cut into more than {HISTOGRAM_BINS_MAX} "
                 f"(HISTOGRAM_BINS_MAX); the edges cover the "
                 f"{HISTOGRAM_WINDOW_PERCENTILES[0]:g}th to "
                 f"{HISTOGRAM_WINDOW_PERCENTILES[1]:g}th percentile of the "
                 "values pooled over the frames, and the values outside are "
                 "counted in n_below and n_above",)
        low, high = span_low, span_high
    first = math.floor(low / width)
    last = max(math.ceil(high / width), first + 1)
    edges = width * np.arange(first, last + 1, dtype=np.float64)
    if edges[-1] < high:
        edges = np.append(edges, edges[-1] + width)
    return edges, notes


def _containers(obj, prefix: str = "") -> dict[str, object]:
    """Every md_stats container held in a result's fields, by name."""
    out: dict[str, object] = {}

    def visit(value, where: str) -> None:
        if isinstance(value, (Distribution, Histogram, Series, Scalar)):
            name = value.name
            key = name if name not in out else f"{name} [{where}]"
            out[key] = value
        elif isinstance(value, Mapping):
            for k, v in value.items():
                visit(v, f"{where} {k}")
        elif isinstance(value, (list, tuple)):
            for k, v in enumerate(value):
                visit(v, f"{where} {k}")

    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            visit(getattr(obj, f.name), f.name)
    else:
        visit(obj, prefix)
    return out


def _option_parameters(prefix: str, options) -> dict[str, object]:
    """An options dataclass as provenance method parameters."""
    out = {}
    for f in dataclasses.fields(options):
        value = getattr(options, f.name)
        if f.name in ("correlation", "measured", "measured_fractions"):
            continue
        out[f"{prefix}.{f.name}"] = _parameter_value(value)
    return out


def _parameter_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return tuple(f"{'-'.join(k) if isinstance(k, tuple) else k}="
                     f"{_parameter_value(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple, frozenset, set)):
        items = sorted(value) if isinstance(value, (frozenset, set)) else value
        return tuple(_parameter_value(v) if not isinstance(v, tuple)
                     else "-".join(str(x) for x in v) for v in items)
    return str(value)


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def analyse(trajectory: Trajectory, request: AnalysisRequest, *,
            progress: Callable[[int, int, str], None] | None = None,
            cancelled: Callable[[], bool] | None = None) -> ModelResult:
    """Run every requested analysis on the chosen frames of ``trajectory``.

    The request is checked first (:func:`missing_inputs`, then against the
    first chosen frame); every missing input is named in one
    :class:`RequestError` before any frame is analysed. Per frame, one pair
    search feeds every per-frame analysis (module docstring); a second pass
    runs only when a cutoff comes from a frame-averaged g(r). An analysis
    that refuses a frame (a ValueError from its module) leaves that frame
    out with the reason, and one that refuses every frame is reported with
    its ``error``; the others go on. Qt-free; ``progress`` and ``cancelled``
    are plain callables for a caller's worker thread.
    """
    if not isinstance(trajectory, Trajectory):
        raise ValueError(f"a Trajectory is needed, not "
                         f"{type(trajectory).__name__}")
    if not isinstance(request, AnalysisRequest):
        raise ValueError("request needs an AnalysisRequest")
    if not request.analyses:
        raise ValueError("the request names no analysis")
    missing = missing_inputs(request)
    if missing:
        raise RequestError(missing)
    return _Run(trajectory, request, progress, cancelled).execute()


@dataclass
class _FrameContext:
    """What one frame gives the consumers: the frame, its valence table, its
    results and bonds at v_bond (None when no analysis reads them), and the
    restricted pair copies of the current pass, by sink name."""

    k: int
    frame: Frame
    table: object
    results: object
    bonds: object
    copies: Mapping[str, list]


class _Run:
    """The state of one :func:`analyse` call."""

    def __init__(self, trajectory: Trajectory, request: AnalysisRequest,
                 progress, cancelled):
        self.t = trajectory
        self.req = request
        self.a = set(request.analyses)
        self._progress = progress
        self._cancel_hook = cancelled
        self._cancelled = False
        self.params = request.params or bv.DEFAULT
        self.notes: list[str] = []
        self.timings: dict[str, float] = {}
        self.searches: dict[int, list[int]] = {}
        self.skipped: dict[int, str] = {}
        self.used: list[int] = []
        self.phase = 1
        self.sinks: dict[int, list[_Sink]] = {1: [], 2: []}
        self.need_r = {1: 0.0, 2: 0.0}
        self.searched_r = {1: 0.0, 2: 0.0}
        self.need_v: dict[int, float | None] = {1: 0.0, 2: None}
        self.p1_done = 0              # frames through pass 1 (progress)
        self.p2_done = 0              # frames through pass 2 (progress)
        self.extra_steps = 0
        self.total_steps = 1
        self.model_notes: list[str] = []
        # per frame: the estimated and the missing bond-valence pairs of its
        # valence table, summed into each bond-based provenance
        self.pair_tallies: dict[int, tuple[dict, dict]] = {}
        self.dynamics_frames: tuple[int, ...] = ()
        self.minima: dict = {}
        self.glass_cutoffs: dict = {}
        self.glass_sources: dict = {}
        self.pass2_wanted = False
        self.pass2_seen: set[int] = set()
        self.glass_result = None
        self.outputs: dict[str, AnalysisOutput] = {}
        self.voronoi_cells: dict[int, object] = {}
        self.search_seconds = 0.0

    # -- progress and cancel --------------------------------------------------
    def is_cancelled(self) -> bool:
        """True once the caller's ``cancelled()`` has returned True."""
        if not self._cancelled and self._cancel_hook is not None \
                and self._cancel_hook():
            self._cancelled = True
        return self._cancelled

    def report(self, stage: str) -> None:
        if self._progress is not None:
            units = (self.p1_done * PROGRESS_UNITS_PASS1
                     + self.p2_done * PROGRESS_UNITS_PASS2 + self.extra_steps)
            self._progress(min(units, self.total_steps), self.total_steps,
                           stage)

    def pass1_progress(self, done: int) -> None:
        """``done`` frames through pass 1 (never fewer than reported)."""
        self.p1_done = max(self.p1_done, min(done, len(self.frames)))

    def pass2_progress(self, done: int) -> None:
        if self.pass2_possible:
            self.p2_done = max(self.p2_done, min(done, len(self.frames)))

    # -- the whole run --------------------------------------------------------
    def execute(self) -> ModelResult:
        start = time.perf_counter()
        self.frames = _chosen_frames(self.t, self.req.frames)
        self.setup()
        self.timings["setup (first frame, checks)"] = time.perf_counter() - start
        steps_after = len([n for n in self.req.analyses
                           if n in TRAJECTORY_ANALYSES + POST_ANALYSES]) + 1
        n = len(self.frames)
        self.frame_total = (n * PROGRESS_UNITS_PASS1
                            + (n * PROGRESS_UNITS_PASS2
                               if self.pass2_possible else 0)
                            if self.has_frame_work else 0)
        self.total_steps = self.frame_total + steps_after
        self.report("starting")
        clock = time.perf_counter()
        if self.has_frame_work:
            if "glass" in self.a:
                self.run_glass()
            else:
                self.own_pass1()
            if not self.used:
                if self._cancelled:
                    raise AnalysisCancelled("cancelled before any frame was "
                                            "analysed")
                reasons = "; ".join(f"frame {k}: {r}" for k, r in
                                    sorted(self.skipped.items()))
                if self.skipped and all(r.startswith("could not be read")
                                        for r in self.skipped.values()):
                    raise FramesUnreadable("no chosen frame could be read: "
                                           + reasons)
                raise ValueError("no frame could be analysed: " + reasons)
            if self.phase == 1:
                self.prepare_pass2()
            if self.pass2_wanted and not self.pass2_seen \
                    and not self._cancelled:
                self.own_pass2()
            self.timings["frames: pair searches and per-frame analyses"] = \
                time.perf_counter() - clock
            self.timings["of which the shared pair searches and their "
                         "copies"] = self.search_seconds
        if self.has_frame_work:
            self.pass1_progress(n)
            self.pass2_progress(n)
        self.report("averaging the per-frame analyses")
        clock = time.perf_counter()
        self.finish_frame_analyses()
        self.timings["averaging the per-frame analyses"] = \
            time.perf_counter() - clock
        self.extra_steps += 1
        self.report("per-frame analyses averaged")
        self.run_trajectory_analyses()
        self.run_post_analyses()
        if self._cancelled and not any(out.ok for out in self.outputs.values()):
            raise AnalysisCancelled("cancelled before any analysis was "
                                    "computed")
        self.timings["total"] = time.perf_counter() - start
        result = self.assemble()
        self.extra_steps = steps_after
        self.report("done")
        return result

    def setup(self) -> None:
        """Read the first chosen frame; fix what every frame shares."""
        first, unreadable = _first_readable(self.t, self.frames)
        self.first = first
        self.first_k = next(k for k in self.frames if k not in unreadable)
        self.species = tuple(first.species)
        self.counts = dict(first.composition)
        try:
            self.ox = model_oxidation(self.species, self.req.ox_overrides)
        except ValueError as error:
            raise RequestError([MissingInput("request", "ox_overrides",
                                             str(error))]) from None
        self.ox_atom = self.ox.per_atom(first.elements)
        # what the first frame shows about the inputs, whatever the analyses
        # (glass states the same in its own result): the net charge with
        # these states and the file's own charges
        self.model_notes.extend(glass.composition(first, self.ox).notes)
        self.cations = tuple(s for s in self.species if self.ox.ox[s] >= 0)
        self.anions = tuple(s for s in self.species if self.ox.ox[s] < 0)
        self.r_bv = bulk.search_radius_ang(first.elements, self.ox_atom,
                                           self.params, self.req.v_list_vu)
        self.bridging = (self.anions if self.req.bridging_anions is None
                         else tuple(self.req.bridging_anions))
        self.formers = self.req.named_formers
        problems = self.check_against_model()
        if problems:
            raise RequestError(problems)
        self.build_consumers()
        self.plan_pass1()

    def check_against_model(self) -> list[MissingInput]:
        """What the first frame shows the request cannot do."""
        req, out = self.req, []
        present = set(self.species)
        if self.formers is not None:
            anionic = sorted(set(self.formers) & set(self.anions))
            if anionic:
                out.append(MissingInput(
                    "formers", "formers", f"{', '.join(anionic)} are anions of "
                    "this model (negative oxidation state); formers are "
                    "cations"))
            absent = sorted(set(self.formers) - present)
            readers = [a for a in FORMER_READERS if a in self.a]
            if absent and readers and not set(self.formers) & present:
                out.append(MissingInput(
                    "/".join(readers), "formers",
                    f"none of the formers named ({', '.join(absent)}) is in "
                    f"the model, which holds {', '.join(self.species)}; every "
                    "anion would count as free and every former-dependent "
                    "descriptor would be empty"))
            elif absent:
                self.model_notes.append(
                    f"formers: {', '.join(absent)} named but not in the model "
                    f"(it holds {', '.join(self.species)}); the former-"
                    "dependent descriptors count "
                    + ", ".join(sorted(set(self.formers) & present)))
        out.extend(self.absent_elements())
        out.extend(_trajectory_problems(self.t, req, self.frames, self.first,
                                        self.a))
        if "scattering" in self.a and req.scattering.r_max_ang is not None:
            dr = float(req.scattering.dr_ang)
            grid = np.arange(dr, float(req.scattering.r_max_ang) + 0.5 * dr, dr)
            reach = float(grid[-1] + dr)
            half = float(self.first.perpendicular_widths_ang.min()) / 2.0
            if reach > half:
                out.append(MissingInput(
                    "scattering", "scattering.r_max_ang",
                    f"r_max + dr = {reach:.6g} Å lies beyond half the smallest "
                    f"perpendicular width of the box ({half:.6g} Å in frame "
                    f"{self.first_k}); there an atom meets its own periodic "
                    "images. Leaving r_max_ang out takes the largest grid "
                    "the boxes hold"))
        stray = sorted(set(self.bridging) - set(self.anions))
        if stray:
            out.append(MissingInput("bridging_anions", "bridging_anions",
                                    f"{', '.join(stray)} are not anions of "
                                    "this model"))
        if self.a & {"exafs", "feff"}:
            absorber = validate_symbol(str(req.exafs.absorber))
            if absorber not in present:
                out.append(MissingInput("exafs", "exafs.absorber",
                                        f"the model holds no {absorber}"))
        if self.a & {"empty-spheres", "free-volume"}:
            radii = req.voids.radii
            if isinstance(radii, Mapping):
                lacking = sorted(present - {validate_symbol(str(k))
                                            for k in radii})
                if lacking:
                    out.append(MissingInput(
                        "empty-spheres/free-volume", "voids.radii",
                        f"no radius for {', '.join(lacking)}"))
            elif radii == "vdw":
                try:
                    md_order.vdw_radii_ang(self.species)
                except ValueError as error:
                    out.append(MissingInput("empty-spheres/free-volume",
                                            "voids.radii", str(error)))
            elif radii not in RADII_NAMES:
                out.append(MissingInput("empty-spheres/free-volume",
                                        "voids.radii", f"{radii!r} is not one "
                                        f"of {RADII_NAMES} or a map"))
        if "conductivity" in self.a:
            try:
                self.conductivity_charges()
            except ValueError as error:
                out.append(MissingInput("conductivity", "charges_e",
                                        str(error)))
        if "nmr" in self.a:
            element = req.nmr.correlation.element
            if element not in present:
                out.append(MissingInput("nmr", "nmr.correlation",
                                        f"the model holds no {element}, the "
                                        "correlation's nucleus"))
        return out

    def absent_elements(self) -> list[MissingInput]:
        """Every element an option of a requested analysis names that the
        model does not hold: an option naming elements is an explicit
        statement, and one the model cannot satisfy would leave its rows
        empty (or end the analysis after every frame)."""
        present = set(self.species)
        net, dyn, x = self.req.network, self.req.dynamics, self.req.exafs
        a = self.a
        checks: list[tuple[str, str, object]] = []
        graph = [n for n in ("rings", "coordination-sequences", "components")
                 if n in a]
        if graph:
            checks.append(("/".join(graph), "network.graph_elements",
                           net.graph_elements))
        for name, option, values in (
                ("rings", "network.ring_t_elements", net.ring_t_elements),
                ("coordination-sequences", "network.cseq_centres",
                 net.cseq_centres),
                ("polyhedral-sharing", "network.sharing_centres",
                 net.sharing_centres),
                ("polyhedral-sharing", "network.sharing_ligands",
                 net.sharing_ligands),
                ("warren-cowley", "network.wc_elements", net.wc_elements)):
            if name in a:
                checks.append((name, option, values))
        timed = [n for n in TRAJECTORY_ANALYSES if n in a]
        if timed:
            checks.append(("/".join(timed), "dynamics.elements", dyn.elements))
        if "distinct-van-hove" in a and dyn.distinct_pairs:
            checks.append(("distinct-van-hove", "dynamics.distinct_pairs",
                           [s for p in dyn.distinct_pairs
                            for s in _pair_key(p)]))
        if "exafs" in a:
            checks.append(("exafs", "exafs.neighbours", x.neighbours))
            checks.append(("exafs", "exafs.first_shell_limits_ang",
                           list(dict(x.first_shell_limits_ang or {}))))
            checks.append(("exafs", "exafs.shells",
                           [shell[0] for shell in x.shells]))
        out = []
        for analysis, option, values in checks:
            if values is None:
                continue
            absent = sorted({validate_symbol(str(v)) for v in values}
                            - present)
            if absent:
                out.append(MissingInput(
                    analysis, option, f"the model holds no "
                    f"{', '.join(absent)} (it holds "
                    f"{', '.join(self.species)})"))
        return out

    def conductivity_charges(self) -> dict[str, float]:
        """The charges of the conductivity, as the request states them."""
        given = self.req.charges_e
        if given == "oxidation states":
            return {s: float(self.ox.ox[s]) for s in self.species}
        if given == "file":
            if self.t.charges_e is None:
                raise ValueError("charges_e 'file': the file states no charge "
                                 "per element (none, or not one per element)")
            return dict(self.t.charges_e)
        out = {}
        for key, value in dict(given).items():
            out[validate_symbol(str(key))] = float(value)
        lacking = sorted(set(self.species) - set(out))
        if lacking:
            raise ValueError(f"no charge for {', '.join(lacking)}")
        return out

    # -- consumers ------------------------------------------------------------
    def build_consumers(self) -> None:
        self.pass1_consumers: list[_Consumer] = []
        self.pass2_consumers: list[_Consumer] = []
        candidates = [(_ScatteringConsumer, {"scattering"}),
                      (_ExafsConsumer, {"exafs"}),
                      (_NmrConsumer, {"nmr"}),
                      (_LifetimeConsumer, {"bond-lifetimes"}),
                      (_VoronoiConsumer, {"voronoi"}),
                      (_OrderConsumer, {"bond-order", "tetrahedral-order",
                                        "polyhedron-shape"}),
                      (_VoidConsumer, {"empty-spheres", "free-volume"}),
                      (_NetworkConsumer, {"rings", "coordination-sequences",
                                          "polyhedral-sharing", "components",
                                          "warren-cowley"})]
        for cls, names in candidates:
            if self.a & names:
                consumer = cls(self)
                self.pass1_consumers.append(consumer)
                if consumer.uses_pass2:
                    self.pass2_consumers.append(consumer)
        # within a frame the costly analyses come last, so that a cancel made
        # during one of them leaves the cheaper ones of that frame done; the
        # order of the list above is kept among equal costs (the Voronoi
        # cells before the order parameters that may read them)
        self.pass1_consumers.sort(key=lambda c: c.cost)
        self.has_frame_work = "glass" in self.a or bool(self.pass1_consumers)

    def plan_pass1(self) -> None:
        """The pass-1 search radius and the copies each consumer keeps."""
        self.needs_table = "glass" in self.a or any(
            c.needs_table for c in self.pass1_consumers)
        r_need = self.r_bv if self.needs_table else 0.0
        v_need = self.r_bv if self.needs_table else 0.0
        for consumer in self.pass1_consumers:
            for name, r_ang, vectors in consumer.sinks(1):
                sink = _Sink(name, r_ang, vectors)
                self.sinks[1].append(sink)
                consumer.sink_of[(1, name)] = sink
                r_need = max(r_need, r_ang)
                if vectors:
                    v_need = max(v_need, r_ang)
        self.need_r[1] = r_need
        self.need_v[1] = v_need
        self.needs_search = r_need > 0.0
        self.pass2_possible = any(c.may_need_pass2()
                                  for c in self.pass2_consumers) or (
            "glass" in self.a and not self.glass_one_pass())

    def glass_one_pass(self) -> bool:
        """True when the user gave every cation-anion cutoff of the model."""
        given = self.req.glass.cutoffs_ang
        if not given:
            return False
        pairs = set()
        for key in dict(given):
            x, y = _pair_key(key)
            pairs.add((x, y) if x in self.cations else (y, x))
        return all((c, a) in pairs for c in self.cations for a in self.anions)

    # -- the shared search ----------------------------------------------------
    def search(self, k: int, frame: Frame, r_ang: float,
               vectors_within_ang: float | None):
        """The frame's one search of this pass, as a stream of its blocks;
        each consumer's restricted copy fills as the blocks pass."""
        phase = self.phase
        radius = max(float(r_ang), self.need_r[phase])
        # the radius actually searched (glass may ask beyond the plan, for
        # a g(r) grid to half the box), for the provenance
        self.searched_r[phase] = max(self.searched_r[phase], radius)
        wanted = self.need_v[phase]
        if vectors_within_ang is None or wanted is None:
            vectors = None
        else:
            vectors = max(float(vectors_within_ang), wanted)
        counts = self.searches.setdefault(k, [0, 0])
        counts[phase - 1] += 1
        blocks = bulk.iter_pairs(frame, radius, vectors_within_ang=vectors)
        sinks = self.sinks[phase]
        key = (k, phase, bulk._frame_digest(frame))
        for sink in sinks:
            sink.start(key)

        def stream():
            source = iter(blocks)
            while True:
                clock = time.perf_counter()
                try:
                    block = next(source)
                except StopIteration:
                    self.search_seconds += time.perf_counter() - clock
                    break
                for sink in sinks:
                    sink.add(block)
                self.search_seconds += time.perf_counter() - clock
                yield block
            for sink in sinks:
                sink.complete = True
        return stream()

    def check_copies(self, k: int, frame: Frame) -> None:
        key = (k, self.phase, bulk._frame_digest(frame))
        for sink in self.sinks[self.phase]:
            if sink.key != key or not sink.complete:
                raise RuntimeError(f"the pair copies of frame {k} are not "
                                   "complete (internal)")

    # -- pass 1 ---------------------------------------------------------------
    def glass_rdf_r_max(self) -> float:
        g = self.req.glass
        return self.r_bv if g.rdf_r_max_ang is None else float(g.rdf_r_max_ang)

    def run_glass(self) -> None:
        g = self.req.glass
        clock = time.perf_counter()

        n = len(self.frames)

        def glass_progress(done: int, total: int) -> None:
            # glass counts its frames through pass 1, then through pass 2
            if done <= n:
                self.pass1_progress(done)
                stage = (f"pass 1: frame {done} of {n} (glass and the "
                         "per-frame analyses)")
            else:
                self.pass1_progress(n)
                self.pass2_progress(done - n)
                stage = (f"pass 2: frame {done - n} of {n} (the pairs within "
                         "the first minima)")
            self.report(stage)

        try:
            result = glass.analyse_trajectory(
                self.t, self.ox, formers=self.formers,
                rdf_r_max_ang=self.glass_rdf_r_max(), minimum=g.minimum(),
                bins=g.bins(), params=self.params,
                v_bond_vu=self.req.v_bond_vu, v_list_vu=self.req.v_list_vu,
                frames=list(self.frames), rdf_dr_ang=g.rdf_dr_ang,
                cutoffs_ang=None if g.cutoffs_ang is None
                else _pair_map(g.cutoffs_ang),
                bridging_anions=self.req.bridging_anions,
                oxide_basis=g.oxide_basis, progress=glass_progress,
                cancelled=self.is_cancelled, pair_search=self.search,
                on_frame=self.on_frame, on_minima=self.on_minima,
                on_distance_frame=self.on_distance_frame)
        except glass.AnalysisCancelled:
            self._cancelled = True
            return
        self.glass_result = result
        for k, reason in result.provenance.frames_skipped.items():
            if k not in self.used:
                self.skipped.setdefault(k, reason)
        consumers = sum(c.total_seconds() for c in self.pass1_consumers)
        seconds = max(time.perf_counter() - clock - consumers
                      - self.search_seconds, 0.0)
        searches = sorted({tuple(self.searches.get(k, (0, 0)))
                           for k in result.provenance.frames_used})
        provenance = dataclasses.replace(
            result.provenance, method_parameters={
                **result.provenance.method_parameters,
                "bulk pair searches per frame (pass 1, pass 2), shared":
                    tuple(f"{a}, {b}" for a, b in searches)})
        self.outputs["glass"] = _glass_output(
            dataclasses.replace(result, provenance=provenance), seconds)

    def own_pass1(self) -> None:
        for position, k in enumerate(self.frames):
            if self.is_cancelled():
                for rest in self.frames[position:]:
                    self.skipped[rest] = ("not analysed: the run was "
                                          "cancelled")
                break
            try:
                frame = self.t.frame(k)
            except FrameError as error:
                self.skipped[k] = f"could not be read: {error}"
                self.pass1_progress(position + 1)
                self.report(f"pass 1: frame {position + 1} of "
                            f"{len(self.frames)} (frame {k}) could not be "
                            "read")
                continue
            table = results = bonds = None
            if self.needs_search:
                blocks = self.search(k, frame, self.need_r[1], self.need_v[1])
                if self.needs_table:
                    table = bulk.valence_table(
                        frame, blocks, self.ox_atom, self.params,
                        v_list_vu=self.req.v_list_vu, r_search_ang=self.r_bv)
                    results = bulk.at_threshold(table, self.req.v_bond_vu)
                    bonds = bulk.bonds_at(table, self.req.v_bond_vu)
                else:
                    for _ in blocks:
                        pass
            self.on_frame(k, frame, table, results, bonds)
            self.pass1_progress(position + 1)
            self.report(f"pass 1: frame {position + 1} of {len(self.frames)} "
                        f"(frame {k})")

    def on_frame(self, k: int, frame: Frame, table, results, bonds) -> None:
        """Every pass-1 consumer on one frame (glass calls this back)."""
        if self.needs_search:
            self.check_copies(k, frame)
        if table is not None:
            self.pair_tallies[k] = (dict(table.estimated_pairs),
                                    dict(table.missing_pairs))
        context = _FrameContext(k, frame, table, results, bonds,
                                {s.name: s.blocks for s in self.sinks[1]})
        for consumer in self.pass1_consumers:
            consumer.run_frame(context, 1)
        for sink in self.sinks[1]:
            sink.blocks = []
        self.used.append(k)

    # -- between the passes, and pass 2 ---------------------------------------
    def on_minima(self, minima, cutoffs, sources) -> None:
        """Glass calls this when its first minima are known (before pass 2)."""
        self.minima = dict(minima)
        self.glass_cutoffs = dict(cutoffs)
        self.glass_sources = dict(sources)
        self.prepare_pass2()

    def prepare_pass2(self) -> None:
        """Once pass 1 is complete: each consumer's cutoffs, the pass-2
        radius and the copies."""
        self.phase = 2
        radius = 0.0
        for consumer in self.pass2_consumers:
            consumer.between_passes()
            for name, r_ang, vectors in consumer.sinks(2):
                sink = _Sink(name, r_ang, vectors)
                self.sinks[2].append(sink)
                consumer.sink_of[(2, name)] = sink
                radius = max(radius, r_ang)
        self.need_r[2] = radius
        self.need_v[2] = None
        self.pass2_wanted = bool(self.sinks[2])

    def on_distance_frame(self, k: int, frame: Frame, dbonds) -> None:
        """Every pass-2 consumer on one frame (glass calls this back)."""
        if self.phase != 2:
            return          # glass's one-pass distance bonds: pass 1 itself
        self.pass2_seen.add(k)
        if not self.sinks[2]:
            return
        self.check_copies(k, frame)
        context = _FrameContext(k, frame, None, None, None,
                                {s.name: s.blocks for s in self.sinks[2]})
        for consumer in self.pass2_consumers:
            consumer.run_frame(context, 2)
        for sink in self.sinks[2]:
            sink.blocks = []

    def own_pass2(self) -> None:
        order = sorted(self.used)
        for position, k in enumerate(order):
            if self.is_cancelled():
                self.notes.append(
                    "cancelled in pass 2: the analyses that read pass-2 "
                    "pairs leave out frames "
                    + _compact([x for x in order if x not in self.pass2_seen]))
                break
            try:
                frame = self.t.frame(k)
            except FrameError as error:
                self.notes.append(f"frame {k} was read in pass 1 but not in "
                                  f"pass 2 ({error}); the analyses that read "
                                  "pass-2 pairs leave it out")
                continue
            for _ in self.search(k, frame, self.need_r[2], None):
                pass
            self.on_distance_frame(k, frame, None)
            self.pass2_progress(position + 1)
            self.report(f"pass 2: frame {position + 1} of {len(order)} "
                        f"(frame {k})")

    # -- after the frames -----------------------------------------------------
    def finish_frame_analyses(self) -> None:
        for consumer in self.pass1_consumers:
            for name, output in consumer.finish().items():
                self.outputs[name] = output

    def run_trajectory_analyses(self) -> None:
        wanted = [n for n in TRAJECTORY_ANALYSES if n in self.a]
        if wanted:
            _Dynamics(self).execute(wanted)

    def run_post_analyses(self) -> None:
        for name in POST_ANALYSES:
            if name not in self.a:
                continue
            clock = time.perf_counter()
            if self.is_cancelled():
                self.outputs[name] = AnalysisOutput(
                    name, {}, None, error="not run: the run was cancelled")
            else:
                try:
                    output = {"nmr-comparison": self.nmr_comparison,
                              "scattering-comparison":
                                  self.scattering_comparison,
                              "feff": self.feff_export}[name]()
                except (ValueError, FrameError, OSError) as error:
                    output = AnalysisOutput(name, {}, None, error=str(error))
                self.outputs[name] = dataclasses.replace(
                    output, seconds=time.perf_counter() - clock)
            self.extra_steps += 1
            self.report(name)

    def nmr_comparison(self) -> AnalysisOutput:
        result = self.glass_result
        if result is None:
            raise ValueError("the glass analysis produced no result to "
                             "compare with")
        tables: dict[str, object] = {}
        raw = []
        for spec in self.req.nmr.measured_fractions:
            found = False
            for definition, section in (("BV", result.bv),
                                        ("distance", result.distance)):
                model = section.get(spec.model_descriptor)
                if model is None:
                    continue
                found = True
                comparison = md_spectroscopy.compare_fractions(
                    model, spec.measured,
                    model_name=f"{spec.model_descriptor} ({definition})",
                    provenance=result.provenance)
                raw.append(comparison)
                name = (f"{spec.measured.descriptor} vs "
                        f"{spec.model_descriptor} ({definition})")
                tables[name] = Table(name, tuple(comparison.as_rows()),
                                     tuple(comparison.notes))
            if not found:
                names = sorted(set(result.bv) | set(result.distance))
                raise ValueError(f"the glass result holds no descriptor "
                                 f"{spec.model_descriptor!r}; it holds "
                                 + ", ".join(names))
        return AnalysisOutput("nmr-comparison", tables, result.provenance,
                              raw=tuple(raw))

    def scattering_comparison(self) -> AnalysisOutput:
        out = self.outputs.get("scattering")
        if out is None or out.raw is None:
            raise ValueError("the scattering analysis produced no result to "
                             "compare with")
        result = out.raw
        tables: dict[str, object] = {}
        raw = []
        for spec in self.req.scattering.measured:
            if spec.function not in CURVE_FUNCTIONS:
                raise ValueError(f"function {spec.function!r} is not one of "
                                 f"{tuple(CURVE_FUNCTIONS)}")
            attr, axis_name = CURVE_FUNCTIONS[spec.function]
            total = result.totals.get(spec.radiation)
            if total is None:
                raise ValueError(f"no {spec.radiation} total was computed "
                                 f"(radiations: {', '.join(result.totals)})")
            series = getattr(total, attr)
            if series is None:
                # the scattering module's own reason, from its notes
                reasons = [n for n in tuple(result.notes) + tuple(total.notes)
                           if n.startswith(f"{attr} is not computed")]
                raise ValueError(f"{spec.function} {spec.radiation} was not "
                                 "computed: " + (reasons[0] if reasons else
                                                 "the scattering result "
                                                 "holds no such curve"))
            measured = md_scattering.read_measured(spec.path,
                                                   axis_name=axis_name)
            comparison = md_scattering.compare_with_measured(
                series.axis, series.mean, measured.axis_values,
                measured.values, axis_name=axis_name,
                quantity=f"{spec.function} {spec.radiation}",
                axis_range=spec.axis_range, scale=spec.scale,
                measured_source=measured.path,
                model_notes=tuple(series.notes))
            raw.append(comparison)
            unit = "1/Å" if axis_name == "q_inv_ang" else "Å"
            # the measured curve is read as the same function, so in the
            # model curve's unit; the scale is a pure number
            value = series.value_unit
            rows = tuple({"quantity": comparison.quantity,
                          axis_name: float(x), f"measured ({value})": float(m),
                          f"model, scaled ({value})": float(v),
                          f"measured - model ({value})": float(d)}
                         for x, m, v, d in zip(comparison.axis_values,
                                               comparison.measured,
                                               comparison.model,
                                               comparison.difference,
                                               strict=True))
            summary = ({"quantity": comparison.quantity,
                        "R_chi": comparison.r_chi,
                        "scale": comparison.scale,
                        "scale source": comparison.scale_source,
                        "points compared": comparison.n_points,
                        "points outside": comparison.n_outside,
                        f"range low ({unit})": comparison.axis_range[0],
                        f"range high ({unit})": comparison.axis_range[1],
                        f"model step ({unit})": comparison.model_step,
                        f"measured step ({unit})": comparison.measured_step,
                        "R_chi definition": comparison.definition,
                        "measured file": comparison.measured_source},)
            notes = tuple(measured.notes) + tuple(comparison.notes)
            name = f"{spec.function} {spec.radiation} vs {Path(spec.path).name}"
            tables[name] = Table(name, rows, notes)
            tables[name + " (R_chi)"] = Table(name + " (R_chi)", summary,
                                              notes)
        return AnalysisOutput("scattering-comparison", tables,
                              result.provenance, raw=tuple(raw))

    def feff_export(self) -> AnalysisOutput:
        f = self.req.feff
        request = md_spectroscopy.FeffRequest(
            f.out_dir, f.n_clusters, f.seed, f.cluster_radius_ang, f.edge,
            r_path_ang=f.r_path_ang, overwrite=f.overwrite)
        frames = sorted(self.used) if self.used else list(self.frames)
        export = md_spectroscopy.export_feff_inputs(
            self.t, self.req.exafs.absorber, request, frames=frames,
            cancelled=self.is_cancelled)
        name = "FEFF clusters"
        notes = tuple(export.notes) + (
            "the clusters are drawn by md_spectroscopy.export_feff_inputs, "
            "which searches each frame holding a drawn absorber once more, "
            "to the cluster radius with vectors (an export, outside the "
            "shared per-frame search)",)
        return AnalysisOutput("feff", {name: Table(name, tuple(
            export.as_rows()), notes)}, export.provenance, raw=export,
            notes=notes)

    # -- the result -----------------------------------------------------------
    def tallies(self, frames: Iterable[int]) -> tuple[dict, dict]:
        """The estimated and the missing bond-valence pairs of ``frames``,
        each contact count summed over them."""
        estimated: dict[str, int] = {}
        missing: dict[str, int] = {}
        for k in frames:
            if k not in self.pair_tallies:
                continue
            for target, source in zip((estimated, missing),
                                      self.pair_tallies[k], strict=True):
                for pair, count in source.items():
                    target[pair] = target.get(pair, 0) + int(count)
        return estimated, missing

    def assemble(self) -> ModelResult:
        req = self.req
        notes = list(self.notes)
        if self.has_frame_work:
            used = sorted(self.used)
        elif self.dynamics_frames:
            used = sorted(self.dynamics_frames)
        else:
            used = list(self.frames)
            notes.append("no analysis received a frame; the frames listed as "
                         "used are the frames chosen")
        skipped = {k: r for k, r in self.skipped.items() if k not in used}
        searches = {k: tuple(v) for k, v in sorted(self.searches.items())}
        pass1 = sorted({v[0] for v in searches.values()})
        pass2 = sorted({v[1] for v in searches.values()})
        method = {"analyses": tuple(req.analyses),
                  "frames chosen": _compact(self.frames),
                  "bulk pair searches per frame, pass 1":
                      ", ".join(str(v) for v in pass1) or "0",
                  "bulk pair searches per frame, pass 2":
                      ", ".join(str(v) for v in pass2) or "0",
                  "pass 1 search radius (Å)": (self.searched_r[1]
                                               or self.need_r[1] or None),
                  "pass 2 search radius (Å)": (self.searched_r[2]
                                               or (self.need_r[2]
                                                   if self.pass2_wanted
                                                   else None)) or None,
                  "bridging anions": self.bridging,
                  "formers stated": ("none" if req.formers == NO_FORMERS
                                     else "not given" if req.formers is None
                                     else "given"),
                  "time axis": ("timestep_fs" if req.timestep_fs is not None
                                else "frame_interval_ps"
                                if req.frame_interval_ps is not None
                                else "the file's frame times"),
                  "timestep_fs": req.timestep_fs,
                  "frame_interval_ps": req.frame_interval_ps,
                  "temperature_k": req.temperature_k,
                  "charges_e": _parameter_value(req.charges_e),
                  "bond-valence estimated parameters (allow_estimated)":
                      bool(self.params.allow_estimated)}
        for group in ("glass", "scattering", "network", "order", "voids",
                      "nmr", "exafs", "feff", "dynamics"):
            method.update(_option_parameters(group, getattr(req, group)))
        notes.extend(self.model_notes)
        estimated, missing = self.tallies(self.used)
        if estimated or missing:
            notes.append("the estimated and missing bond-valence pair counts "
                         "are summed over the frames used")
        if self._cancelled:
            notes.append(f"the run was cancelled: {len(self.used)} of "
                         f"{len(self.frames)} chosen frames were analysed in "
                         "pass 1; each analysis is over the frames it "
                         "received")
        outputs = {name: self.outputs[name] for name in ANALYSES
                   if name in self.outputs}
        for name in req.analyses:
            if name not in outputs:
                outputs[name] = AnalysisOutput(
                    name, {}, None, error="not run" + (
                        ": the run was cancelled" if self._cancelled else ""))
        for name, out in outputs.items():
            if out.error:
                notes.append(f"{name}: not computed: {out.error}")
        for (a, b), found in sorted(self.minima.items()):
            if a <= b:
                notes.append(
                    f"g(r) first minimum {a}-{b}: "
                    + (f"{found.r_ang!r} Å ({found.source}; {found.method})"
                       if found.found else f"none ({found.reason})"))
        provenance = Provenance.from_trajectory(
            self.t, frames_used=used, frames_skipped=skipped,
            params=self.params, ox=self.ox, v_bond_vu=req.v_bond_vu,
            v_list_vu=req.v_list_vu, r_search_ang=self.r_bv,
            formers=self.formers, cutoffs_ang=dict(self.glass_cutoffs),
            cutoff_sources=dict(self.glass_sources),
            method_parameters=method, notes=tuple(notes),
            estimated_pairs=estimated, missing_pairs=missing)
        timings = dict(self.timings)
        for name, out in outputs.items():
            timings[f"analysis {name}"] = out.seconds
        return ModelResult(request=req, provenance=provenance, outputs=outputs,
                           searches=searches, minima=dict(self.minima),
                           notes=tuple(notes), cancelled=self._cancelled,
                           timings_s=timings,
                           model_notes=tuple(dict.fromkeys(self.model_notes)))

    def provenance(self, frames_used: Sequence[int],
                   frames_skipped: Mapping[int, str], *, method: Mapping,
                   notes: Sequence[str] = (), bonds: bool = False,
                   formers: bool = False, cutoffs: Mapping | None = None,
                   sources: Mapping | None = None) -> Provenance | None:
        """The provenance of one analysis computed here. ``bonds``: it read
        the bonds at v_bond, so it states the bond-valence set, both
        thresholds, whether the estimator was on, and the estimated and
        missing bond-valence pairs of its frames."""
        if not frames_used:
            return None
        searches = sorted({tuple(self.searches.get(k, (0, 0)))
                           for k in frames_used})
        method = dict(method)
        method["bulk pair searches per frame (pass 1, pass 2), shared"] = \
            tuple(f"{a}, {b}" for a, b in searches)
        fields_ = dict(ox=self.ox, formers=self.formers if formers else None,
                       cutoffs_ang=dict(cutoffs or {}),
                       cutoff_sources=dict(sources or {}),
                       method_parameters=method)
        notes = list(notes)
        if bonds:
            estimated, missing = self.tallies(frames_used)
            method["bond-valence estimated parameters (allow_estimated)"] = \
                bool(self.params.allow_estimated)
            fields_.update(params=self.params, v_bond_vu=self.req.v_bond_vu,
                           v_list_vu=self.req.v_list_vu,
                           r_search_ang=self.r_bv, estimated_pairs=estimated,
                           missing_pairs=missing)
            if estimated or missing:
                notes.append("the estimated and missing bond-valence pair "
                             "counts are summed over the frames used")
        return Provenance.from_trajectory(
            self.t, frames_used=sorted(frames_used),
            frames_skipped={k: r for k, r in frames_skipped.items()
                            if k not in frames_used},
            notes=tuple(notes), **fields_)

    def auto_cutoffs(self, pairs: Iterable[tuple[str, str]]
                     ) -> tuple[dict, dict, list[str]]:
        """The glass first minima of ``pairs`` as cutoffs, with their
        sources, and a note for each pair without one."""
        cutoffs, sources, notes = {}, {}, []
        for a, b in pairs:
            found = self.minima.get((a, b))
            if found is None or not found.found:
                reason = ("the glass analysis located none" if found is None
                          else found.reason)
                notes.append(f"no cutoff for {a}-{b}: {reason}")
                continue
            cutoffs[(a, b)] = float(found.r_ang)
            if found.source == "user":
                sources[(a, b)] = "user (glass.cutoffs_ang)"
            else:
                low, high = found.floor_r_ang or (found.r_ang, found.r_ang)
                sources[(a, b)] = (f"first minimum of the frame-averaged "
                                   f"g_{a}{b}(r) of the glass analysis, "
                                   f"{found.method}; its floor runs "
                                   f"{low:.6g}-{high:.6g} Å")
        return cutoffs, sources, notes


class _Consumer:
    """One or more per-frame analyses fed from the shared search."""

    names: tuple[str, ...] = ()
    cost = 0
    uses_pass2 = False

    def __init__(self, run: _Run):
        self.run = run
        self.req = run.req
        self.needs_table = False
        self.used: dict[str, list[int]] = {n: [] for n in self.names}
        self.skipped: dict[str, dict[int, str]] = {n: {} for n in self.names}
        self.seconds: dict[str, float] = {n: 0.0 for n in self.names}
        self.sink_of: dict[tuple[int, str], _Sink] = {}

    # what the consumer asks of the search ----------------------------------------
    def sinks(self, phase: int) -> list[tuple[str, float, bool]]:
        """(name, radius in Å, vectors?) of each copy kept in this pass."""
        return []

    def may_need_pass2(self) -> bool:
        return False

    def between_passes(self) -> None:
        """Called once pass 1 is complete, before pass 2 is planned."""

    def active(self, name: str) -> bool:
        return name in self.run.a

    def total_seconds(self) -> float:
        return math.fsum(self.seconds.values())

    # per frame -----------------------------------------------------------------
    def run_frame(self, ctx: _FrameContext, phase: int) -> None:
        self.frame(ctx, phase)

    def frame(self, ctx: _FrameContext, phase: int) -> None:
        raise NotImplementedError

    def attempt(self, name: str, k: int, work: Callable[[], None], *,
                record: bool = True) -> bool:
        """Run one analysis on frame k; a refusal leaves the frame out of it
        with the reason. ``record`` False leaves the frame list alone (a
        second pass of an analysis whose frame was recorded in pass 1)."""
        if not self.active(name):
            return False
        if self.run.is_cancelled():
            self.skipped[name][k] = "not analysed: the run was cancelled"
            return False
        clock = time.perf_counter()
        try:
            work()
        except md_network.Cancelled:
            self.run._cancelled = True
            self.skipped[name][k] = "not analysed: the run was cancelled"
            return False
        except (ValueError, FrameError) as error:
            self.skipped[name][k] = str(error)
            return False
        finally:
            self.seconds[name] += time.perf_counter() - clock
        if record:
            self.used[name].append(k)
        return True

    # at the end ------------------------------------------------------------------
    def finish(self) -> dict[str, AnalysisOutput]:
        raise NotImplementedError

    def output(self, name: str, tables: Mapping, provenance, *, raw=None,
               notes: Sequence[str] = ()) -> AnalysisOutput:
        if not self.used[name]:
            reasons = "; ".join(f"frame {k}: {r}" for k, r in
                                sorted(self.skipped[name].items())[:3])
            if not reasons and self.run._cancelled:
                # an analysis of pass 2 that the cancel stopped before it
                return AnalysisOutput(
                    name, {}, None, error="not run: the run was cancelled "
                    "before this analysis received a frame",
                    seconds=self.seconds[name])
            return AnalysisOutput(name, {}, None,
                                  error="no frame was analysed"
                                  + (f" ({reasons})" if reasons else ""),
                                  seconds=self.seconds[name])
        return AnalysisOutput(name, dict(tables), provenance, raw=raw,
                              notes=tuple(notes), seconds=self.seconds[name])

    def failed(self, name: str, error: Exception) -> AnalysisOutput:
        return AnalysisOutput(name, {}, None, error=str(error),
                              seconds=self.seconds[name])

    def skipped_with_run(self, name: str) -> dict[int, str]:
        """This analysis's skipped frames, with the run's own."""
        out = dict(self.run.skipped)
        for k in self.run.used:
            if k not in self.used[name] and k not in self.skipped[name]:
                out[k] = ("not analysed: pass 2 did not reach this frame"
                          if self.uses_pass2 else "not analysed")
        out.update(self.skipped[name])
        return out

    def timed(self, name: str, work: Callable[[], object]):
        clock = time.perf_counter()
        try:
            return work()
        finally:
            self.seconds[name] += time.perf_counter() - clock


# -- scattering ---------------------------------------------------------------

class _ScatteringConsumer(_Consumer):
    """md_scattering on partials from the frame's shared search.

    With ``r_max_ang`` given, every frame is on that grid and a frame whose
    box cannot hold it is refused (md_scattering.frame_partials). Without
    it, each frame's partials run to the largest grid its own box holds, at
    most the first frame's (which sets the shared search), and every frame
    is cut to the shortest of the frames used before averaging (module
    docstring), so the average does not depend on the frame order.
    """

    names = ("scattering",)
    cost = 1

    def __init__(self, run: _Run):
        super().__init__(run)
        o = self.req.scattering
        self.dr_ang = float(o.dr_ang)
        self.given_r_max = o.r_max_ang is not None
        if self.given_r_max:
            self.r_max_ang = float(o.r_max_ang)
        else:
            r_max = self.largest_r_max(run.first, self.dr_ang)
            if r_max is None:
                half = float(run.first.perpendicular_widths_ang.min()) / 2.0
                raise RequestError([MissingInput(
                    "scattering", "scattering.dr_ang", f"the box (half its "
                    f"smallest width {half:.6g} Å) holds no g(r) grid at dr "
                    f"{self.dr_ang:g} Å")])
            self.r_max_ang = r_max
        grid = _r_grid(self.r_max_ang, self.dr_ang)
        self.reach_ang = float(grid[-1] + self.dr_ang)
        step = float(o.q_step_inv_ang)
        self.q_inv_ang = np.arange(step, float(o.q_grid_max_inv_ang)
                                   + 0.5 * step, step)
        self.partials: dict[int, md_scattering.FramePartials] = {}
        self.frame_r_max: dict[int, float] = {}

    @staticmethod
    def largest_r_max(frame: Frame, dr_ang: float) -> float | None:
        """The largest r_max = n dr whose grid's last point plus dr lies
        within half the smallest perpendicular box width (frame_partials's
        limit); None when the box holds no grid of three points."""
        half = float(frame.perpendicular_widths_ang.min()) / 2.0
        n = int(math.floor(half / dr_ang)) - 1
        while n >= 2:
            r_max = n * dr_ang
            if _r_grid(r_max, dr_ang)[-1] + dr_ang <= half:
                return float(r_max)
            n -= 1
        return None

    def sinks(self, phase):
        return [("scattering", self.reach_ang, False)] if phase == 1 else []

    def frame(self, ctx, phase):
        sink = self.sink_of[(1, "scattering")]

        def work():
            r_max = self.r_max_ang
            if not self.given_r_max:
                own = self.largest_r_max(ctx.frame, self.dr_ang)
                if own is None:
                    raise ValueError(
                        f"the box of frame {ctx.k} (half its smallest "
                        f"perpendicular width "
                        f"{float(ctx.frame.perpendicular_widths_ang.min()) / 2:.6g}"
                        f" Å) holds no g(r) grid at dr {self.dr_ang:g} Å")
                r_max = min(r_max, own)
            self.partials[ctx.k] = md_scattering.frame_partials(
                ctx.frame, sink.blocks, r_max_ang=r_max, dr_ang=self.dr_ang)
            self.frame_r_max[ctx.k] = r_max
        self.attempt("scattering", ctx.k, work)

    def finish(self):
        name = "scattering"
        if not self.partials:
            return {name: self.output(name, {}, None)}
        o = self.req.scattering
        used = sorted(self.partials)
        r_max = min(self.frame_r_max[k] for k in used)
        size = _r_grid(r_max, self.dr_ang).size
        partials = {k: _trimmed_partials(self.partials[k], size)
                    for k in used}
        cut_notes = []
        if any(self.partials[k].r_ang.size != size for k in used):
            cut_notes.append(
                f"the box changes between the frames used: the r grid stops "
                f"at r_max = {r_max!r} Å, the largest grid the smallest box "
                "holds, and the partials of the larger boxes are cut there "
                "before averaging, so the average does not depend on the "
                "order of the frames")
        # the G(r) from S(Q) starts at the first Q of the grid unless a
        # Qmin is given; 0 would lie below the S(Q) (module docstring)
        q_min = o.termination_q_min_inv_ang
        q_min_from = "scattering.termination_q_min_inv_ang"
        if q_min is None:
            if o.termination_q_max_inv_ang is None:
                q_min, q_min_from = 0.0, "no termination"
            else:
                q_min = float(self.q_inv_ang[0])
                q_min_from = ("the first Q of the grid "
                              "(scattering.q_step_inv_ang)")
        try:
            result = self.timed(name, lambda: md_scattering.analyse_trajectory(
                self.run.t, r_max_ang=r_max, dr_ang=self.dr_ang,
                q_inv_ang=self.q_inv_ang, r_window=o.r_window,
                radiations=tuple(o.radiations), frames=used,
                q_max_inv_ang=o.termination_q_max_inv_ang,
                q_min_inv_ang=q_min,
                q_window=o.q_window,
                fsdp_window_inv_ang=None if o.fsdp_window_inv_ang is None
                else tuple(o.fsdp_window_inv_ang),
                fsdp_baseline=o.fsdp_baseline,
                neutron_lengths_fm=o.neutron_lengths_fm,
                lengths_source=o.lengths_source,
                given_partials=partials))
        except ValueError as error:
            return {name: self.failed(name, error)}
        self.partials, self.frame_r_max = {}, {}
        provenance = result.provenance
        extra = {k: r for k, r in self.skipped_with_run(name).items()
                 if k not in provenance.frames_used}
        searches = sorted({tuple(self.run.searches.get(k, (0, 0)))
                           for k in provenance.frames_used})
        provenance = dataclasses.replace(
            provenance,
            frames_skipped={**provenance.frames_skipped, **extra},
            notes=tuple(provenance.notes) + tuple(cut_notes),
            method_parameters={
                **provenance.method_parameters,
                "r_max_ang from": ("the largest grid the smallest box of the "
                                   "frames used holds" if not self.given_r_max
                                   else "scattering.r_max_ang"),
                "termination_q_min_inv_ang from": q_min_from,
                "bulk pair searches per frame (pass 1, pass 2), shared":
                    tuple(f"{a}, {b}" for a, b in searches)})
        tables: dict[str, object] = {"number density": result.rho0}
        for series in result.partial_g.values():
            tables[series.name] = series
        for series in result.partial_s.values():
            tables[series.name] = series
        for radiation, total in result.totals.items():
            for attr in ("s", "f_reduced", "f_keen", "pdf_g", "keen_g",
                         "keen_d", "keen_t", "pdf_g_from_sq"):
                series = getattr(total, attr)
                if series is not None:
                    tables[series.name] = series
            if total.fsdp:
                for scalar in total.fsdp.values():
                    tables[scalar.name] = scalar
            weights = f"Faber-Ziman weights at Q = 0, {radiation}"
            tables[weights] = Table(weights, tuple(
                {"pair": f"{a}-{b}", "weight w_ab (1)": w,
                 f"<b>^2 at Q = 0 ({total.weight_unit})": total.b_mean_sq_q0}
                for (a, b), w in total.weights_q0.items()), tuple(total.notes))
        if result.bhatia_thornton:
            for series in result.bhatia_thornton.values():
                tables[series.name] = series
        return {name: AnalysisOutput(name, tables, provenance, raw=result,
                                     notes=tuple(result.notes)
                                     + tuple(cut_notes),
                                     seconds=self.seconds[name])}


def _r_grid(r_max_ang: float, dr_ang: float) -> np.ndarray:
    """pdf.pair_distribution's grid, arange(dr, r_max + dr/2, dr), as
    md_scattering builds it."""
    return np.arange(dr_ang, r_max_ang + 0.5 * dr_ang, dr_ang)


def _trimmed_partials(partials: md_scattering.FramePartials, size: int
                      ) -> md_scattering.FramePartials:
    """``partials`` on the first ``size`` points of its r grid.

    np.arange gives point i as dr + i dr whatever the stop, so the first
    ``size`` points are the grid of the shorter r_max; a deposit lands on
    the grid points either side of its distance, and the counts on those
    points below the cut are the same whether the grid continues or not.
    The cut partials therefore equal ``frame_partials`` run to the shorter
    r_max (``tests/test_md_analysis.py`` checks it exactly).
    """
    if partials.r_ang.size == size:
        return partials

    def cut(values: Mapping) -> dict:
        return {key: _frozen(array[:size]) for key, array in values.items()}

    return dataclasses.replace(
        partials, r_ang=_frozen(partials.r_ang[:size]),
        pair_hist=cut(partials.pair_hist), g=cut(partials.g),
        n_cum=cut(partials.n_cum))


# -- EXAFS --------------------------------------------------------------------

class _ExafsConsumer(_Consumer):
    """``md_spectroscopy.md_exafs``, its frames fed from the shared search.

    The assembly follows md_exafs (the limits, the shells, their notes and
    the provenance keys), with its pair searches replaced by restricted
    copies of each frame's one search; ``tests/test_md_analysis.py``
    compares the two on the same frames.
    """

    names = ("exafs",)
    cost = 1
    uses_pass2 = True

    def __init__(self, run: _Run):
        super().__init__(run)
        o = self.req.exafs
        self.absorber = validate_symbol(str(o.absorber))
        self.r_max_ang = (run.glass_rdf_r_max() if o.r_max_ang is None
                          else float(o.r_max_ang))
        self.dr_ang = float(o.dr_ang)
        grid = np.arange(self.dr_ang, self.r_max_ang + 0.5 * self.dr_ang,
                         self.dr_ang)
        self.grid_need_ang = float(grid[-1] + self.dr_ang)
        self.weighting = o.weighting
        self.minimum = self.req.glass.minimum()
        self.given = {validate_symbol(str(e)): float(v) for e, v in
                      dict(o.first_shell_limits_ang or {}).items()}
        self.extra = [(validate_symbol(str(e)), float(lo), float(hi))
                      for e, lo, hi in o.shells]
        present = set(run.species)
        self.chosen = (sorted(present) if o.neighbours is None
                       else [validate_symbol(str(s)) for s in o.neighbours])
        unknown = sorted((set(self.chosen) | set(self.given)
                          | {e for e, _, _ in self.extra}) - present)
        if unknown:
            raise RequestError([MissingInput(
                "exafs", "exafs.neighbours", f"the model holds no "
                f"{', '.join(unknown)}")])
        unused = sorted(set(self.given) - set(self.chosen))
        if unused:
            raise RequestError([MissingInput(
                "exafs", "exafs.first_shell_limits_ang", f"a limit is given "
                f"for {', '.join(unused)}, which is not among the neighbours "
                f"({', '.join(self.chosen)})")])
        self.known = all(e in self.given for e in self.chosen)
        self.r_known = (max([self.given.get(e, 0.0) for e in self.chosen]
                            + [h for _, _, h in self.extra] + [0.0])
                        if self.known else 0.0)
        self.r_base = max(self.grid_need_ang, self.r_known)
        half = float(run.first.perpendicular_widths_ang.min() / 2.0)
        for what, value in ([("r_max_ang", self.r_max_ang)]
                            + [(f"the first-shell limit of {e}", v)
                               for e, v in self.given.items()]
                            + [(f"the outer edge of the {e} shell", h)
                               for e, _, h in self.extra]):
            if value > half:
                raise RequestError([MissingInput(
                    "exafs", "exafs", f"{what} {value} is beyond half the "
                    f"smallest perpendicular width of the box ({half!r} Å)")])
        self.specs = ([(e, 0.0, self.given[e], "user") for e in self.chosen
                       if e in self.given]
                      + [(e, lo, hi, "user") for e, lo, hi in self.extra]) \
            if self.known else []
        self.shells: list[dict[int, object]] = [{} for _ in self.specs]
        self.g_rows = {e: {} for e in self.chosen}
        self.n_rows = {e: {} for e in self.chosen}
        self.volumes: dict[int, float] = {}
        self.pair_counts: dict[str, tuple[int, int]] = {}
        self.grid = None
        self.limits: dict = {}
        self.r_second = None
        self.pass2_done: list[int] = []
        self.pass2_reasons: dict[int, str] = {}

    def sinks(self, phase):
        if phase == 1:
            return [("exafs", self.r_base, False)]
        return [("exafs", self.r_second, False)] if self.r_second else []

    def may_need_pass2(self) -> bool:
        return not self.known

    def frame(self, ctx, phase):
        if phase == 1:
            sink = self.sink_of[(1, "exafs")]

            def work():
                ap = md_spectroscopy.absorber_pairs(ctx.frame, sink.blocks,
                                                    self.absorber)
                rdfs = md_spectroscopy.absorber_rdf(
                    ap, r_max_ang=self.r_max_ang, dr_ang=self.dr_ang,
                    neighbours=self.chosen)
                for element, rdf in rdfs.items():
                    counts = (rdf.n_absorbers, rdf.n_neighbours)
                    if self.pair_counts.setdefault(element, counts) != counts:
                        raise ValueError(f"frame {ctx.k} holds other numbers "
                                         f"of {self.absorber} or {element} "
                                         "atoms than the first frame")
                shells = [md_spectroscopy.shell_cumulants(
                    ap, spec[0], r_lo_ang=spec[1], r_hi_ang=spec[2],
                    weighting=self.weighting) for spec in self.specs]
                for element, rdf in rdfs.items():
                    self.g_rows[element][ctx.k] = rdf.g
                    self.n_rows[element][ctx.k] = rdf.n_cum
                    self.grid = rdf.r_ang
                self.volumes[ctx.k] = ap.volume_ang3
                for c, shell in enumerate(shells):
                    self.shells[c][ctx.k] = shell
            self.attempt("exafs", ctx.k, work)
            return
        if self.known or not self.r_second or \
                ctx.k not in self.used["exafs"]:
            return
        sink = self.sink_of[(2, "exafs")]

        def work2():
            ap = md_spectroscopy.absorber_pairs(ctx.frame, sink.blocks,
                                                self.absorber)
            shells = [md_spectroscopy.shell_cumulants(
                ap, spec[0], r_lo_ang=spec[1], r_hi_ang=spec[2],
                weighting=self.weighting) for spec in self.specs]
            for c, shell in enumerate(shells):
                self.shells[c][ctx.k] = shell
            self.pass2_done.append(ctx.k)
        before = dict(self.skipped["exafs"])
        if not self.attempt("exafs", ctx.k, work2, record=False):
            # a pass-2 refusal leaves the frame's g(r) in, its shells out
            reason = self.skipped["exafs"].pop(ctx.k, "not analysed")
            self.skipped["exafs"].update(before)
            self.pass2_reasons[ctx.k] = reason

    def series(self, used: list[int]):
        g_series, n_series = {}, {}
        for element in self.chosen:
            name = f"{self.absorber}-{element}"
            g_series[element] = Series.from_frames(
                self.grid, [self.g_rows[element][k] for k in used],
                name=f"g {name}(r)", axis_name="r_ang", axis_unit="Å",
                value_unit="1", frames=used)
            n_series[element] = Series.from_frames(
                self.grid, [self.n_rows[element][k] for k in used],
                name=f"N {name}(r)", axis_name="r_ang", axis_unit="Å",
                value_unit="1", frames=used,
                notes=(f"the exact number of {element} atoms within r of a "
                       f"{self.absorber} atom, per {self.absorber} atom",))
        return g_series, n_series

    def between_passes(self) -> None:
        """The first-shell limits from the frame-averaged g(r), as md_exafs
        locates them, and the pass-2 shells they give."""
        used = sorted(self.used["exafs"])
        if not used:
            return
        self.g_series, self.n_series = self.series(used)
        for element in self.chosen:
            if element in self.given:
                self.limits[element] = md_spectroscopy.ShellLimit(
                    self.absorber, element, self.given[element], None, None,
                    "user", "first-shell limit given by the user")
                continue
            n_a, n_b = self.pair_counts[element]
            same = element == self.absorber
            if same and n_a < 2:
                self.limits[element] = md_spectroscopy.ShellLimit(
                    self.absorber, element, None, None, None, "auto",
                    self.minimum.describe(), reason=f"the model holds one "
                    f"{self.absorber} atom, so it has no {self.absorber}-"
                    f"{self.absorber} pair")
                continue
            volumes = [self.volumes[k] for k in used]
            per_unit = glass.pair_counts_per_unit_g(
                self.grid, self.dr_ang, n_a, n_b,
                math.fsum(volumes) / len(volumes), same_element=same,
                n_frames=len(used))
            self.limits[element] = md_spectroscopy.first_minimum(
                self.grid, self.g_series[element].mean, method=self.minimum,
                absorber=self.absorber, neighbour=element,
                level=(n_a - 1) / n_a if same else 1.0,
                pairs_per_unit_g=per_unit)
        if not self.known:
            auto = "first minimum of the frame-averaged g(r)"
            self.specs = ([(e, 0.0, self.limits[e].r_ang, auto)
                           for e in self.chosen
                           if self.limits[e].r_ang is not None]
                          + [(e, lo, hi, "user") for e, lo, hi in self.extra])
            self.shells = [{} for _ in self.specs]
            if self.specs:
                self.r_second = max(s[2] for s in self.specs)

    def finish(self):
        name = "exafs"
        used = sorted(self.used[name])
        if not used:
            return {name: self.output(name, {}, None)}
        if not self.limits:
            self.between_passes()
        notes = []
        for element, limit in self.limits.items():
            if limit.r_ang is None:
                notes.append(f"{self.absorber}-{element}: no first-shell limit "
                             f"({limit.reason}); no first shell is reported "
                             "for it")
            notes.extend(limit.notes)
        searched = f"to {self.r_base!r} Å"
        if self.known:
            shell_frames = used
            notes.append(f"every shell limit was known in advance: the shells "
                         f"and g(r) come from each frame's shared pair search "
                         f"(pairs {searched})")
            if self.given:
                notes.append(f"a first-minimum method was given "
                             f"({self.minimum.rule} rule) but every "
                             "first-shell limit was given too, so no limit "
                             "was located with it and it is not in the "
                             "provenance")
        else:
            shell_frames = sorted(self.pass2_done)
            for e, lo, hi in self.extra:
                if lo == 0.0 and self.limits[e].r_ang == hi and \
                        e not in self.given:
                    notes.append(f"the user shell {e} (0, {hi:g}] Å equals the "
                                 f"automatic first shell of {e}; both are "
                                 "listed")
            if self.specs:
                notes.append(
                    "the first-shell limits came from the frame-averaged "
                    "g(r), known only after every frame was read, so each "
                    f"frame was searched twice: pass 1 (pairs {searched}) "
                    f"for g(r), pass 2 (pairs to {self.r_second!r} Å) for "
                    "the shell distances")
                left = sorted(set(used) - set(shell_frames))
                if left:
                    notes.append("the shells leave out frames "
                                 f"{_compact(left)}: "
                                 + "; ".join(f"frame {k}: {r}" for k, r in
                                             sorted(self.pass2_reasons.items())
                                             [:3]) if self.pass2_reasons
                                 else "the shells leave out frames "
                                 f"{_compact(left)} (pass 2 did not reach "
                                 "them)")
            else:
                notes.append("no first-shell limit was found and no shell was "
                             "given, so no shell was measured")
        averaged = [md_spectroscopy.average_shells(
            [shells[k] for k in shell_frames], frames=shell_frames,
            limit_source=spec[3])
            for spec, shells in zip(self.specs, self.shells, strict=True)] \
            if shell_frames else []
        method_parameters = {"r_max_ang": self.r_max_ang,
                             "dr_ang": self.dr_ang,
                             "weighting": self.weighting,
                             "user shells": tuple(
                                 f"{e} ({lo:g}, {hi:g}] Å"
                                 for e, lo, hi in self.extra),
                             "pass 1 search radius (Å)": self.r_base}
        if not self.known:
            m = self.minimum
            method_parameters.update({
                "first minimum": m.describe(),
                "first minimum rule": m.rule,
                "first minimum smooth_sigma_ang": m.smooth_sigma_ang,
                "first minimum flat_rule": m.flat_rule,
                "first minimum margin_std_errors": m.margin_std_errors})
        if self.r_second is not None:
            method_parameters["pass 2 search radius (Å)"] = self.r_second
        cutoffs = {(self.absorber, e): self.limits[e].r_ang
                   for e in self.chosen if self.limits[e].r_ang is not None}
        sources = {(self.absorber, e): (self.limits[e].source
                                        if self.limits[e].source == "user"
                                        else self.limits[e].method)
                   for e in self.chosen if self.limits[e].r_ang is not None}
        provenance = self.run.provenance(
            used, self.skipped_with_run(name), method=method_parameters,
            notes=notes, cutoffs=cutoffs, sources=sources)
        result = md_spectroscopy.MdExafs(
            absorber=self.absorber, g=self.g_series, n_cum=self.n_series,
            limits=self.limits, shells=tuple(averaged), feff=None,
            provenance=provenance, searches_per_frame=1 if self.known else 2,
            notes=tuple(notes))
        tables: dict[str, object] = {}
        for element in self.chosen:
            tables[self.g_series[element].name] = self.g_series[element]
            tables[self.n_series[element].name] = self.n_series[element]
        sheets = result.as_sheets()
        tables["shell limits"] = Table("shell limits",
                                       tuple(sheets["shell limits"]))
        tables["shells"] = Table(
            "shells", tuple(sheets["shells"]),
            tuple(n for s in averaged for n in s.notes),
            frames=None if list(shell_frames) == list(used)
            else tuple(shell_frames))
        return {name: self.output(name, tables, provenance, raw=result,
                                  notes=notes)}


# -- NMR ----------------------------------------------------------------------

class _NmrConsumer(_Consumer):
    """``md_spectroscopy.md_nmr`` on the bonds of the shared search."""

    names = ("nmr",)

    def __init__(self, run: _Run):
        super().__init__(run)
        self.needs_table = True
        self.predictions: dict[int, object] = {}

    def frame(self, ctx, phase):
        def work():
            o = self.req.nmr
            self.predictions[ctx.k] = md_spectroscopy.predict_shifts(
                o.correlation, ctx.frame, ctx.bonds, formers=self.run.formers)
        self.attempt("nmr", ctx.k, work)

    def finish(self):
        name = "nmr"
        used = sorted(self.used[name])
        if not used:
            return {name: self.output(name, {}, None)}
        o = self.req.nmr
        first, last, step = (float(v) for v in o.delta_ppm)
        grid = np.arange(first, last + 0.5 * step, step)
        notes: list[str] = []
        # the provenance of an analysis on bonds carries the estimated and
        # missing bond-valence pairs of its frames (_Run.provenance)
        provenance = self.run.provenance(
            used, self.skipped_with_run(name), bonds=True, formers=True,
            method={"lineshape": o.lineshape, "fwhm_ppm": float(o.fwhm_ppm),
                    "delta_ppm first": float(grid[0]),
                    "delta_ppm last": float(grid[-1]),
                    "delta_ppm points": int(grid.size),
                    "shift histogram bin (ppm)": step,
                    "correlation": o.correlation.describe(),
                    "correlation reference": o.correlation.reference},
            notes=notes)

        def work():
            predictions = [self.predictions[k] for k in used]
            spectrum = md_spectroscopy.nmr_spectrum(
                predictions, grid, fwhm_ppm=float(o.fwhm_ppm),
                lineshape=o.lineshape, frames=used, provenance=provenance)
            shifts = [np.asarray(p.shift_ppm, dtype=np.float64)
                      for p in predictions]
            edges, edge_notes = _edges_over(shifts, step)
            histogram = md_spectroscopy.shift_histogram(
                predictions, edges, frames=used).with_notes(
                f"bin width {step:g} ppm (the step of nmr.delta_ppm), edges "
                "over the shifts predicted", *edge_notes)
            return spectrum, histogram
        try:
            spectrum, histogram = self.timed(name, work)
        except ValueError as error:
            return {name: self.failed(name, error)}
        sheets = spectrum.as_sheets()
        tables = {spectrum.spectrum.name: spectrum.spectrum,
                  spectrum.outside.name: spectrum.outside,
                  spectrum.counts.name: spectrum.counts,
                  spectrum.mean_shift.name: spectrum.mean_shift,
                  "spectrum areas": Table("spectrum areas",
                                          tuple(sheets["areas"])),
                  histogram.name: histogram}
        return {name: self.output(name, tables, provenance, raw=spectrum,
                                  notes=tuple(spectrum.notes) + tuple(notes))}


# -- bond lifetimes -----------------------------------------------------------

class _LifetimeConsumer(_Consumer):
    """Bond lifetimes from the bonds of the shared search
    (``md_dynamics.bond_timeline``, the route for bonds already computed)."""

    names = ("bond-lifetimes",)

    def __init__(self, run: _Run):
        super().__init__(run)
        self.needs_table = True
        self.bonds: dict[int, bulk.Bonds] = {}

    def frame(self, ctx, phase):
        def work():
            b = ctx.bonds
            n = len(b)
            # the timeline reads the rows, the images and the threshold; the
            # vectors, distances and valences are not kept for every frame
            self.bonds[ctx.k] = bulk.Bonds(
                cation=b.cation, anion=b.anion, image=b.image,
                vec_ang=np.broadcast_to(np.zeros(3), (n, 3)),
                d_ang=np.broadcast_to(np.zeros(1), (n,)),
                v_vu=np.broadcast_to(np.zeros(1), (n,)), v_bond_vu=b.v_bond_vu)
        self.attempt("bond-lifetimes", ctx.k, work)

    def finish(self):
        name = "bond-lifetimes"
        used = sorted(self.used[name])
        if not used:
            return {name: self.output(name, {}, None)}
        req, dyn = self.req, self.req.dynamics

        def work():
            t_ps, source, notes = md_dynamics.time_axis_ps(
                self.run.t, used, timestep_fs=req.timestep_fs,
                frame_interval_ps=req.frame_interval_ps)
            notes = list(notes) + [f"reader: {n}" for n in self.run.t.notes]
            notes.append(f"bonds: cation-anion contacts with v > "
                         f"{req.v_bond_vu:g} v.u. (bulk.bonds_at), from each "
                         "frame's shared pair search")
            timeline = md_dynamics.bond_timeline(
                [self.bonds[k] for k in used],
                elements=self.run.first.elements, t_ps=t_ps, frames=used,
                source_path=self.run.t.source_path, time_source=source,
                notes=notes)
            lifetimes = md_dynamics.bond_lifetimes(
                timeline, max_lag_t_ps=dyn.lifetime_max_lag_t_ps,
                n_blocks=dyn.n_blocks,
                gap_tolerance_frames=dyn.gap_tolerance_frames)
            residence = None
            if dyn.residence_method is not None:
                residence = md_dynamics.residence_time(
                    lifetimes, method=dyn.residence_method,
                    t_min_ps=dyn.residence_t_min_ps,
                    t_max_ps=dyn.residence_t_max_ps)
            return source, timeline, lifetimes, residence
        try:
            source, timeline, lifetimes, residence = self.timed(name, work)
        except ValueError as error:
            return {name: self.failed(name, error)}
        self.bonds = {}
        tables = _containers(lifetimes)
        if residence:
            rows = tuple(row for r in residence.values() for row in r.as_rows())
            tables["residence times"] = Table(
                "residence times", rows,
                tuple(n for r in residence.values() for n in r.notes))
        method = dict(lifetimes.method_parameters)
        method["time source"] = source
        provenance = self.run.provenance(
            used, self.skipped_with_run(name), bonds=True, method=method,
            notes=tuple(lifetimes.notes))
        return {name: self.output(name, tables, provenance,
                                  raw=(timeline, lifetimes, residence),
                                  notes=tuple(lifetimes.notes))}


# -- Voronoi cells ------------------------------------------------------------

class _VoronoiConsumer(_Consumer):
    """Voronoi cells of each frame (they search nothing)."""

    names = ("voronoi",)
    cost = 2

    def __init__(self, run: _Run):
        super().__init__(run)
        self.cells: dict[int, md_order.VoronoiCells] = {}

    def frame(self, ctx, phase):
        o = self.req.order

        def work():
            cells = md_order.voronoi_cells(
                ctx.frame, min_face_area_ang2=o.min_face_area_ang2,
                min_edge_ang=o.min_edge_ang)
            # the order parameters of this frame may read the full cells
            self.run.voronoi_cells = {ctx.k: cells}
            empty = np.zeros(0)
            # the face table is not kept for every frame: the averages read
            # the per-atom counts, the volumes and the thresholds
            self.cells[ctx.k] = dataclasses.replace(
                cells, face_atom=empty, face_nbr=empty, face_image=empty,
                face_distance_ang=empty, face_area_ang2=empty,
                face_edges=empty, face_counted=empty)
        self.attempt("voronoi", ctx.k, work)

    def finish(self):
        name = "voronoi"
        used = sorted(self.used[name])
        self.run.voronoi_cells = {}
        if not used:
            return {name: self.output(name, {}, None)}
        cells = [self.cells[k] for k in used]

        def work():
            tables: dict[str, object] = {}
            for element in (None,) + tuple(self.run.species):
                d = md_order.voronoi_index_distribution(cells, element=element,
                                                        frames=used)
                tables[d.name] = d
                d = md_order.voronoi_cn_distribution(cells, element=element,
                                                     frames=used)
                tables[d.name] = d
            width = self.req.order.cell_volume_bin_ang3
            per_frame = [(c.elements, c.volume_ang3) for c in cells]
            edges, edge_notes = _edges_over([v for _, v in per_frame], width)
            for h in md_order.element_histograms(
                    per_frame, edges, name="Voronoi cell volume", unit="Å^3",
                    frames=used, notes=(f"bin width {width:g} Å^3 "
                                        "(order.cell_volume_bin_ang3)",)
                    + edge_notes).values():
                tables[h.name] = h
            return tables
        try:
            tables = self.timed(name, work)
        except ValueError as error:
            return {name: self.failed(name, error)}
        notes = sorted({n for c in cells for n in c.notes})
        provenance = self.run.provenance(
            used, self.skipped_with_run(name),
            method=dict(cells[0].method_parameters), notes=notes)
        return {name: self.output(name, tables, provenance, notes=notes)}


# -- local order --------------------------------------------------------------

class _OrderConsumer(_Consumer):
    """Steinhardt order, q_tet and polyhedron shape on one neighbour list."""

    names = ("bond-order", "tetrahedral-order", "polyhedron-shape")
    cost = 3
    uses_pass2 = True

    _SPECS = {
        "q": ("q_l", "1", "q_bin"), "w_hat": ("w-hat_l", "1", "w_bin"),
        "q_bar": ("q-bar_l", "1", "q_bin"),
        "w_bar_hat": ("w-bar-hat_l", "1", "w_bin"),
        "q_tet": ("q_tet", "1", "q_tet_bin"),
        "baur": ("Baur distortion index", "1", "baur_bin"),
        "angle_variance_deg2": ("bond-angle variance", "deg^2",
                                "angle_variance_bin_deg2"),
        "quadratic_elongation": ("quadratic elongation", "1",
                                 "elongation_bin"),
        "volume_ang3": ("polyhedron volume", "Å^3", "volume_bin_ang3"),
        "ecn": ("effective coordination number ECoN", "1", "ecn_bin"),
    }
    _GROUPS = {"tetrahedral-order": ("q_tet",),
               "polyhedron-shape": ("baur", "angle_variance_deg2",
                                    "quadratic_elongation", "volume_ang3",
                                    "ecn")}

    def __init__(self, run: _Run):
        super().__init__(run)
        o = self.req.order
        if o.neighbours not in NEIGHBOUR_KINDS:
            raise RequestError([MissingInput(
                "bond-order", "order.neighbours", f"{o.neighbours!r} is not "
                f"one of {NEIGHBOUR_KINDS}")])
        self.kind = o.neighbours
        self.needs_table = self.kind == "bond valence"
        self.user = None if o.distance_cutoffs_ang is None else \
            _pair_map(o.distance_cutoffs_ang)
        self.cutoffs: dict | None = None
        self.sources: dict | None = None
        if self.kind == "distance" and self.user is not None:
            self.cutoffs, self.sources = {}, {}
            for (a, b), v in self.user.items():
                for key in ((a, b), (b, a)):
                    self.cutoffs.setdefault(key, v)
                    self.sources.setdefault(
                        key, "user (order.distance_cutoffs_ang)")
        self.values: dict[str, dict[int, tuple]] = {}
        self.global_q: dict[int, dict] = {}
        self.definitions: set[str] = set()
        self.cut_notes: list[str] = []

    def work_phase(self) -> int:
        return 2 if (self.kind == "distance" and self.user is None) else 1

    def sinks(self, phase):
        if self.kind != "distance" or phase != self.work_phase() \
                or not self.cutoffs:
            return []
        return [("order", max(self.cutoffs.values()), True)]

    def may_need_pass2(self) -> bool:
        return self.work_phase() == 2

    def between_passes(self) -> None:
        if self.work_phase() != 2:
            return
        pairs = [(c, a) for c in self.run.cations for a in self.run.anions]
        cutoffs, sources, notes = self.run.auto_cutoffs(pairs)
        self.cutoffs, self.sources = {}, {}
        for (a, b), v in cutoffs.items():
            for key in ((a, b), (b, a)):
                self.cutoffs[key] = v
                self.sources[key] = sources[(a, b)]
        self.cut_notes = notes

    def neighbours(self, ctx, phase):
        o = self.req.order
        if self.kind == "bond valence":
            return md_order.neighbours_from_bonds(ctx.frame, ctx.bonds)
        if self.kind == "voronoi":
            cells = self.run.voronoi_cells.get(ctx.k)
            if cells is None:
                cells = md_order.voronoi_cells(
                    ctx.frame, min_face_area_ang2=o.min_face_area_ang2,
                    min_edge_ang=o.min_edge_ang)
            return md_order.neighbours_from_voronoi(ctx.frame, cells)
        sink = self.sink_of[(phase, "order")]
        return md_order.neighbours_from_pairs(ctx.frame, sink.blocks,
                                              self.cutoffs, self.sources)

    def frame(self, ctx, phase):
        if phase != self.work_phase():
            return
        if self.kind == "distance" and not self.cutoffs:
            return
        if phase == 2 and ctx.k not in self.run.used:
            return
        k = ctx.k
        holder = {}
        active = [n for n in self.names if self.active(n)]
        if self.run.is_cancelled():
            for name in active:
                self.skipped[name][k] = "not analysed: the run was cancelled"
            return
        clock = time.perf_counter()
        try:
            holder["nb"] = self.neighbours(ctx, phase)
        except (ValueError, FrameError) as error:
            for name in active:
                self.skipped[name][k] = str(error)
            return
        finally:
            for name in active:
                self.seconds[name] += (time.perf_counter() - clock) / len(active)
        nb = holder["nb"]
        self.definitions.add(nb.definition)
        o = self.req.order

        def bond_order():
            result = md_order.steinhardt(nb, o.degrees)
            for field_name in result.FIELDS:
                for degree in result.degrees:
                    self.values.setdefault(f"{field_name}_{degree}", {})[k] = (
                        result.elements, result.values(field_name, degree),
                        result.notes)
            q = {f"global Q_{degree}": float(v) for degree, v in zip(
                result.degrees, result.global_q, strict=True)}
            q.update({f"global W-hat_{degree}": float(v) for degree, v in zip(
                result.degrees, result.global_w_hat, strict=True)})
            self.global_q[k] = q

        def tetra():
            result = md_order.tetrahedral_order(nb,
                                                selection=o.q_tet_selection)
            self.values.setdefault("q_tet", {})[k] = (
                result.elements, result.q_tet, result.notes)

        def shape():
            result = md_order.polyhedron_shape(nb)
            for key in self._GROUPS["polyhedron-shape"]:
                self.values.setdefault(key, {})[k] = (
                    result.elements, getattr(result, key), result.notes)
        self.attempt("bond-order", k, bond_order)
        self.attempt("tetrahedral-order", k, tetra)
        self.attempt("polyhedron-shape", k, shape)

    def keys_of(self, name: str) -> list[str]:
        if name == "bond-order":
            return [key for key in self.values
                    if key.rsplit("_", 1)[-1].isdigit()]
        return [key for key in self._GROUPS[name] if key in self.values]

    def histograms(self, key: str, used: list[int]) -> dict[str, Histogram]:
        if key.rsplit("_", 1)[-1].isdigit():
            base, degree = key.rsplit("_", 1)
            label, unit, width_name = self._SPECS[base]
            label = label.replace("_l", f"_{degree}")
        else:
            label, unit, width_name = self._SPECS[key]
        width = getattr(self.req.order, width_name)
        rows = self.values[key]
        per_frame = [(rows[k][0], rows[k][1]) for k in used]
        edges, edge_notes = _edges_over([v for _, v in per_frame], width)
        notes = sorted(self.definitions) + [
            f"bin width {width:g} {unit} (order.{width_name}), edges over "
            "the values measured"] + list(edge_notes)
        return md_order.element_histograms(per_frame, edges, name=label,
                                           unit=unit, frames=used, notes=notes)

    def finish(self):
        out = {}
        for name in self.names:
            if not self.active(name):
                continue
            used = sorted(self.used[name])
            if not used:
                out[name] = self.output(name, {}, None)
                continue

            def work(name=name, used=used):
                tables: dict[str, object] = {}
                for key in self.keys_of(name):
                    for h in self.histograms(key, used).values():
                        tables[h.name] = h
                if name == "bond-order":
                    for key in self.global_q[used[0]]:
                        tables[key] = Scalar(
                            key, "1", [self.global_q[k][key] for k in used],
                            frames=used, notes=tuple(sorted(self.definitions)))
                return tables
            try:
                tables = self.timed(name, work)
            except ValueError as error:
                out[name] = self.failed(name, error)
                continue
            frame_notes = sorted({n for key in self.keys_of(name)
                                  for k in used
                                  for n in self.values[key][k][2]})
            provenance = self.run.provenance(
                used, self.skipped_with_run(name),
                bonds=self.kind == "bond valence",
                method=_option_parameters("order", self.req.order),
                notes=tuple(sorted(self.definitions)) + tuple(self.cut_notes)
                + tuple(frame_notes),
                cutoffs=dict(self.cutoffs or {}),
                sources=dict(self.sources or {}))
            out[name] = self.output(name, tables, provenance,
                                    notes=tuple(frame_notes))
        return out


# -- voids --------------------------------------------------------------------

class _VoidConsumer(_Consumer):
    """Empty spheres and free volume (they search nothing)."""

    names = ("empty-spheres", "free-volume")
    cost = 4

    def __init__(self, run: _Run):
        super().__init__(run)
        v = self.req.voids
        if isinstance(v.radii, Mapping):
            self.radii = {validate_symbol(str(key)): float(r)
                          for key, r in v.radii.items()}
            self.radii_source = str(v.radii_source)
        elif v.radii == "zero":
            self.radii = {s: 0.0 for s in run.species}
            self.radii_source = "radii of 0 (the Delaunay circumspheres)"
        else:
            self.radii = dict(md_order.vdw_radii_ang(tuple(run.species)))
            self.radii_source = md_order.VDW_RADII_SOURCE
        self.spheres: dict[int, object] = {}
        self.volumes: dict[int, object] = {}

    def frame(self, ctx, phase):
        v = self.req.voids

        def spheres():
            result = md_order.empty_spheres(ctx.frame, self.radii,
                                            radii_source=self.radii_source)
            # only the radii are averaged; the tetrahedra are not kept
            self.spheres[ctx.k] = dataclasses.replace(
                result, vertices=np.zeros((0, 4), np.int64),
                vertex_image=np.zeros((0, 4, 3), np.int64),
                centre_frac=np.zeros((0, 3)), circumradius_ang=np.zeros(0),
                volume_ang3=np.zeros(0))

        def free():
            self.volumes[ctx.k] = md_order.free_volume(
                ctx.frame, self.radii, radii_source=self.radii_source,
                probe_radius_ang=float(v.probe_radius_ang),
                grid_spacing_ang=float(v.grid_spacing_ang))
        self.attempt("empty-spheres", ctx.k, spheres)
        self.attempt("free-volume", ctx.k, free)

    def finish(self):
        out = {}
        method = _option_parameters("voids", self.req.voids)
        method["radii (Å)"] = tuple(f"{s}={r:g}" for s, r in self.radii.items())
        method["radii source"] = self.radii_source
        name = "empty-spheres"
        if self.active(name):
            used = sorted(self.used[name])
            if not used:
                out[name] = self.output(name, {}, None)
            else:
                spheres = [self.spheres[k] for k in used]
                width = self.req.voids.sphere_bin_ang
                try:
                    edges, edge_notes = _edges_over(
                        [s.radius_ang for s in spheres], width)
                    h = self.timed(name, lambda: md_order.empty_sphere_histogram(
                        spheres, edges, frames=used).with_notes(
                        f"bin width {width:g} Å (voids.sphere_bin_ang), "
                        "edges over the radii measured", *edge_notes))
                    notes = sorted({n for s in spheres for n in s.notes})
                    provenance = self.run.provenance(
                        used, self.skipped_with_run(name), method=method,
                        notes=notes)
                    out[name] = self.output(name, {h.name: h}, provenance,
                                            notes=notes)
                except ValueError as error:
                    out[name] = self.failed(name, error)
        name = "free-volume"
        if self.active(name):
            used = sorted(self.used[name])
            if not used:
                out[name] = self.output(name, {}, None)
            else:
                results = [self.volumes[k] for k in used]
                try:
                    scalars = md_order.free_volume_scalars(results,
                                                           frames=used)
                    notes = sorted({n for r in results for n in r.notes})
                    provenance = self.run.provenance(
                        used, self.skipped_with_run(name), method=method,
                        notes=notes)
                    out[name] = self.output(
                        name, {s.name: s for s in scalars.values()},
                        provenance, notes=notes)
                except ValueError as error:
                    out[name] = self.failed(name, error)
        return out


# -- the network --------------------------------------------------------------

class _NetworkConsumer(_Consumer):
    """The md_network analyses, on graphs built from the frame's bonds or
    from its shared search."""

    names = ("rings", "coordination-sequences", "polyhedral-sharing",
             "components", "warren-cowley")
    cost = 5
    uses_pass2 = True
    _GRAPH_USERS = ("components", "coordination-sequences", "rings")

    def __init__(self, run: _Run):
        super().__init__(run)
        o = self.req.network
        if o.graph not in GRAPH_KINDS:
            raise RequestError([MissingInput("rings", "network.graph",
                                             f"{o.graph!r} is not one of "
                                             f"{GRAPH_KINDS}")])
        self.graph_kind = o.graph
        formers = tuple(sorted(run.formers or ()))
        self.formers = formers
        self.graph_elements = (None if o.graph_elements is None
                               else _symbols_of(o.graph_elements))
        if self.graph_kind == "bond valence" and self.graph_elements is None:
            self.graph_elements = formers + tuple(run.bridging)
        self.t_elements = (formers if o.ring_t_elements is None
                           else _symbols_of(o.ring_t_elements))
        self.cseq_centres = (formers if o.cseq_centres is None
                             else _symbols_of(o.cseq_centres))
        self.sharing_centres = (formers if o.sharing_centres is None
                                else _symbols_of(o.sharing_centres))
        self.sharing_ligands = (tuple(run.anions) if o.sharing_ligands is None
                                else _symbols_of(o.sharing_ligands))
        self.wc_elements = (tuple(run.species) if o.wc_elements is None
                            else _symbols_of(o.wc_elements))
        self.graph_wanted = any(self.active(n) for n in self._GRAPH_USERS)
        self.needs_table = ((self.graph_wanted
                             and self.graph_kind != "distance")
                            or self.active("polyhedral-sharing"))
        self.graph_user = None if o.distance_cutoffs_ang is None else \
            _pair_map(o.distance_cutoffs_ang)
        self.wc_user = None if o.wc_cutoffs_ang is None else \
            _pair_map(o.wc_cutoffs_ang)
        self.graph_cut = self.graph_user
        self.graph_src = None if self.graph_user is None else {
            key: "user (network.distance_cutoffs_ang)"
            for key in self.graph_user}
        self.wc_cut = self.wc_user
        self.wc_src = None if self.wc_user is None else {
            key: "user (network.wc_cutoffs_ang)" for key in self.wc_user}
        self.results: dict[str, dict[int, object]] = {n: {} for n in
                                                      self.names}
        self.cut_notes: list[str] = []

    def graph_phase(self) -> int:
        return 2 if (self.graph_kind == "distance" and self.graph_user is None
                     and self.graph_wanted) else 1

    def wc_phase(self) -> int:
        return 2 if (self.active("warren-cowley")
                     and self.wc_user is None) else 1

    def may_need_pass2(self) -> bool:
        return self.graph_phase() == 2 or self.wc_phase() == 2

    def sinks(self, phase):
        out = []
        if self.graph_wanted and self.graph_kind == "distance" and \
                phase == self.graph_phase() and self.graph_cut:
            out.append(("network graph", max(self.graph_cut.values()), True))
        if self.active("warren-cowley") and phase == self.wc_phase() and \
                self.wc_cut:
            out.append(("warren-cowley", max(self.wc_cut.values()), True))
        return out

    def between_passes(self) -> None:
        """Cutoffs from the glass first minima, once pass 1 is done."""
        if self.graph_phase() == 2:
            nodes = set(self.graph_elements or ())
            pairs = [(c, a) for c in self.run.cations for a in self.run.anions
                     if c in nodes and a in nodes]
            self.graph_cut, self.graph_src, notes = self.run.auto_cutoffs(pairs)
            self.cut_notes += [f"network graph: {n}" for n in notes]
        if self.wc_phase() == 2:
            pairs = [(x, y) for i, x in enumerate(self.wc_elements)
                     for y in self.wc_elements[i:]]
            self.wc_cut, self.wc_src, notes = self.run.auto_cutoffs(pairs)
            self.cut_notes += [f"Warren-Cowley: {n}" for n in notes]

    def main_graph(self, ctx, phase):
        run = self.run
        if self.graph_kind == "bridging anion":
            return md_network.bridged_graph(
                ctx.frame, ctx.table, run.req.v_bond_vu,
                formers=self.formers, anions=run.bridging)
        if self.graph_kind == "bond valence":
            return md_network.graph_from_bonds(
                ctx.frame, ctx.table, run.req.v_bond_vu,
                elements=self.graph_elements)
        sink = self.sink_of[(phase, "network graph")]
        return md_network.graph_from_pairs(
            ctx.frame, sink.blocks, self.graph_cut, self.graph_src,
            elements=self.graph_elements)

    def frame(self, ctx, phase):
        o = self.req.network
        k = ctx.k
        if phase == 2 and k not in self.run.used:
            return
        if phase == 1 and self.active("polyhedral-sharing"):
            def sharing():
                graph = md_network.graph_from_bonds(
                    ctx.frame, ctx.table, self.run.req.v_bond_vu,
                    elements=tuple(sorted(set(self.sharing_centres)
                                          | set(self.sharing_ligands))))
                self.results["polyhedral-sharing"][k] = \
                    md_network.polyhedral_connectivity(
                        graph, centres=self.sharing_centres,
                        ligands=self.sharing_ligands)
            self.attempt("polyhedral-sharing", k, sharing)
        if phase == self.wc_phase() and self.active("warren-cowley") and \
                self.wc_cut:
            def wc():
                sink = self.sink_of[(phase, "warren-cowley")]
                graph = md_network.graph_from_pairs(
                    ctx.frame, sink.blocks, self.wc_cut, self.wc_src,
                    elements=self.wc_elements)
                self.results["warren-cowley"][k] = \
                    md_network.warren_cowley(graph)
            self.attempt("warren-cowley", k, wc)
        wanted = [n for n in self._GRAPH_USERS if self.active(n)]
        if phase != self.graph_phase() or not wanted:
            return
        if self.graph_kind == "distance" and not self.graph_cut:
            return
        holder = {}

        def build():
            holder["graph"] = self.main_graph(ctx, phase)
        if not self.attempt(wanted[0], k, build, record=False):
            reason = self.skipped[wanted[0]].get(k, "not analysed")
            for name in wanted:
                self.skipped[name][k] = reason
            return
        graph = holder["graph"]

        def comps():
            self.results["components"][k] = md_network.components(graph)

        def cseq():
            self.results["coordination-sequences"][k] = \
                md_network.coordination_sequences(
                    graph, o.n_shells, centres=self.cseq_centres,
                    cancelled=self.run.is_cancelled)

        def rings():
            self.results["rings"][k] = md_network.ring_statistics(
                graph, o.ring_criterion, o.ring_max_size,
                t_elements=self.t_elements,
                progress=lambda done, total: self.run.report(
                    f"rings, frame {k}: {done} of {total}"),
                cancelled=self.run.is_cancelled)
        self.attempt("components", k, comps)
        self.attempt("coordination-sequences", k, cseq)
        self.attempt("rings", k, rings)

    def finish(self):
        out = {}
        averagers = {
            "rings": md_network.average_rings,
            "coordination-sequences":
                md_network.average_coordination_sequences,
            "polyhedral-sharing": md_network.average_polyhedral_connectivity,
            "components": md_network.average_components,
            "warren-cowley": md_network.average_warren_cowley}
        for name in self.names:
            if not self.active(name):
                continue
            used = sorted(self.used[name])
            if not used:
                out[name] = self.output(name, {}, None)
                continue
            try:
                averaged = self.timed(name, lambda name=name, used=used:
                                      averagers[name]([self.results[name][k]
                                                       for k in used],
                                                      frames=used))
            except ValueError as error:
                out[name] = self.failed(name, error)
                continue
            tables = {value.name: value for value in averaged.values()}
            first = self.results[name][used[0]]
            definition = getattr(first, "graph_definition", "")
            distance = (name == "warren-cowley"
                        or (self.graph_kind == "distance"
                            and name != "polyhedral-sharing"))
            cut = self.wc_cut if name == "warren-cowley" else self.graph_cut
            src = self.wc_src if name == "warren-cowley" else self.graph_src
            provenance = self.run.provenance(
                used, self.skipped_with_run(name), bonds=not distance,
                formers=True,
                method=_option_parameters("network", self.req.network),
                notes=((definition,) if definition else ())
                + tuple(self.cut_notes),
                cutoffs=dict(cut or {}) if distance else {},
                sources=dict(src or {}) if distance else {})
            out[name] = self.output(name, tables, provenance,
                                    notes=(definition,) if definition else ())
        self.results = {n: {} for n in self.names}
        return out


# -- dynamics -----------------------------------------------------------------

class _Dynamics:
    """The md_dynamics analyses of the whole trajectory, on the unwrapped
    positions of the chosen frames (``md_dynamics.collect_tracks``)."""

    def __init__(self, run: _Run):
        self.run = run
        self.req = run.req
        self._fit_cache = None

    def execute(self, wanted: list[str]) -> None:
        run, req, dyn = self.run, self.req, self.req.dynamics
        if run.is_cancelled():
            # reading every frame's positions is itself a pass over the file
            for name in wanted:
                run.outputs[name] = AnalysisOutput(
                    name, {}, None, error="not run: the run was cancelled")
                run.extra_steps += 1
            return
        clock = time.perf_counter()
        try:
            tracks = md_dynamics.collect_tracks(
                run.t, frames=list(run.frames), timestep_fs=req.timestep_fs,
                frame_interval_ps=req.frame_interval_ps,
                elements=dyn.elements, unwrap=dyn.unwrap,
                step_limit_fraction=dyn.step_limit_fraction)
        except (ValueError, FrameError) as error:
            for name in wanted:
                run.outputs[name] = AnalysisOutput(
                    name, {}, None, error=f"the tracks could not be "
                    f"collected: {error}")
                run.extra_steps += 1
            return
        run.timings["dynamics: reading the unwrapped positions"] = \
            time.perf_counter() - clock
        run.dynamics_frames = tuple(int(k) for k in tracks.frames)
        for name in wanted:
            clock = time.perf_counter()
            if run.is_cancelled():
                run.outputs[name] = AnalysisOutput(
                    name, {}, None, error="not run: the run was cancelled")
                run.extra_steps += 1
                continue
            try:
                tables, raw, method, notes = getattr(
                    self, "a_" + name.replace("-", "_"))(tracks)
                provenance = run.provenance(
                    list(tracks.frames), {}, method={
                        **tracks.method_parameters, **method,
                        "time source": tracks.time_source},
                    notes=tuple(tracks.notes) + tuple(notes))
                output = AnalysisOutput(name, tables, provenance, raw=raw,
                                        notes=tuple(notes))
            except (ValueError, FrameError) as error:
                output = AnalysisOutput(name, {}, None, error=str(error))
            run.outputs[name] = dataclasses.replace(
                output, seconds=time.perf_counter() - clock)
            run.extra_steps += 1
            run.report(name)

    def fits(self, tracks):
        """The MSD and its diffusion fits, computed once for msd and the
        conductivity."""
        if self._fit_cache is None:
            dyn = self.req.dynamics
            result = md_dynamics.msd(tracks,
                                     remove_com_drift=dyn.remove_com_drift,
                                     n_blocks=dyn.n_blocks,
                                     max_lag_t_ps=dyn.max_lag_t_ps)
            fits = None
            if dyn.fit_t_min_ps is not None and dyn.fit_t_max_ps is not None:
                fits = md_dynamics.fit_diffusion(result, dyn.fit_t_min_ps,
                                                 dyn.fit_t_max_ps)
            self._fit_cache = (result, fits)
        return self._fit_cache

    @staticmethod
    def fit_table(fits, name: str) -> Table:
        # each element's fit repeats the notes of the tracks; once each
        return Table(name, tuple(row for f in fits.values()
                                 for row in f.as_rows()),
                     tuple(dict.fromkeys(n for f in fits.values()
                                         for n in f.notes)))

    def a_msd(self, tracks):
        result, fits = self.fits(tracks)
        tables = _containers(result)
        notes = list(result.notes)
        if fits:
            name = "diffusion coefficients (MSD fit)"
            tables[name] = self.fit_table(fits, name)
        else:
            notes.append("no fit window (dynamics.fit_t_min_ps and "
                         "fit_t_max_ps) was given, so no diffusion "
                         "coefficient is reported")
        return tables, (result, fits), dict(result.method_parameters), notes

    def a_self_correlations(self, tracks):
        dyn = self.req.dynamics
        edges = None
        if dyn.van_hove_edges_r_ang is not None:
            first, last, step = dyn.van_hove_edges_r_ang
            edges = np.arange(first, last + 0.5 * step, step)
        result = md_dynamics.self_correlations(
            tracks, list(dyn.lag_t_ps), remove_com_drift=dyn.remove_com_drift,
            edges_r_ang=edges, q_inv_ang=None if dyn.isf_q_inv_ang is None
            else list(dyn.isf_q_inv_ang), n_blocks=dyn.n_blocks)
        return (_containers(result), result, dict(result.method_parameters),
                list(result.notes))

    def a_distinct_van_hove(self, tracks):
        dyn = self.req.dynamics
        first, last, step = dyn.van_hove_edges_r_ang
        edges = np.arange(first, last + 0.5 * step, step)
        result = md_dynamics.distinct_van_hove(
            tracks, list(dyn.lag_t_ps), edges,
            [_pair_key(p) for p in dyn.distinct_pairs],
            n_origins=dyn.distinct_n_origins)
        notes = list(result.notes) + [
            "md_dynamics.distinct_van_hove searches the frames it builds from "
            "the positions at two times (its own pair search, outside the "
            "shared per-frame search)"]
        return (_containers(result), result, dict(result.method_parameters),
                notes)

    def a_vacf(self, tracks):
        dyn = self.req.dynamics
        result = md_dynamics.vacf(tracks, velocities=dyn.velocities,
                                  max_lag_t_ps=dyn.max_lag_t_ps,
                                  n_blocks=dyn.n_blocks)
        spectrum = md_dynamics.vdos(result, window=dyn.vdos_window)
        tables = _containers(result)
        tables.update(_containers(spectrum))
        notes = list(result.notes) + list(spectrum.notes)
        gk = None
        if dyn.green_kubo_t_max_ps is not None:
            gk = md_dynamics.green_kubo_diffusion(result,
                                                  dyn.green_kubo_t_max_ps)
            name = "diffusion coefficients (Green-Kubo)"
            tables[name] = self.fit_table(gk, name)
        method = {**result.method_parameters, **spectrum.method_parameters}
        return tables, (result, spectrum, gk), method, notes

    def a_kinetic_temperature(self, tracks):
        scalar = md_dynamics.kinetic_temperature(tracks)
        return {scalar.name: scalar}, scalar, {}, list(scalar.notes)

    def a_conductivity(self, tracks):
        req, dyn = self.req, self.req.dynamics
        charges = self.run.conductivity_charges()
        result, fits = self.fits(tracks)
        if not fits:
            raise ValueError("the conductivity needs the MSD fit window "
                             "(dynamics.fit_t_min_ps and fit_t_max_ps)")
        ne = md_dynamics.nernst_einstein(
            fits, charges_e=charges, temperature_k=req.temperature_k,
            n_atoms=tracks.composition_model,
            volume_ang3=tracks.mean_volume_ang3)
        collective = md_dynamics.collective_conductivity(
            tracks, charges_e=charges, temperature_k=req.temperature_k,
            t_min_ps=dyn.fit_t_min_ps, t_max_ps=dyn.fit_t_max_ps,
            n_blocks=dyn.n_blocks, max_lag_t_ps=dyn.max_lag_t_ps)
        haven = md_dynamics.haven_ratio(ne, collective)
        name = "diffusion coefficients (MSD fit)"
        tables: dict[str, object] = {
            name: self.fit_table(fits, name),
            "Nernst-Einstein conductivity": Table(
                "Nernst-Einstein conductivity", tuple(ne.as_rows()),
                tuple(ne.notes)),
            "collective conductivity": Table(
                "collective conductivity", tuple(collective.as_rows()),
                tuple(collective.notes)),
            "Haven ratio": Table("Haven ratio", tuple(haven.as_rows()),
                                 tuple(haven.notes))}
        tables.update(_containers(collective))
        notes = list(ne.notes) + list(collective.notes) + list(haven.notes)
        method = {"charges_e": tuple(f"{s}={q:g}" for s, q in charges.items()),
                  "charges source": _parameter_value(req.charges_e),
                  "temperature_k": float(req.temperature_k),
                  "fit window (ps)": (dyn.fit_t_min_ps, dyn.fit_t_max_ps)}
        return tables, (fits, ne, collective, haven), method, notes


# ---------------------------------------------------------------------------
# glass
# ---------------------------------------------------------------------------

def _glass_output(result: glass.GlassResult, seconds: float) -> AnalysisOutput:
    """The glass result as tables: composition, density, minima, cutoffs and
    every descriptor under both definitions."""
    comp = result.composition
    tables: dict[str, object] = {}
    tables["composition"] = Table("composition", tuple(
        {"element": element, "atoms": count,
         "atomic %": comp.atomic_percent[element],
         "oxidation state": comp.ox[element]}
        for element, count in comp.counts.items()), tuple(comp.notes))
    tables["composition summary"] = Table("composition summary", (
        {"atoms": comp.n_atoms, "net charge (e)": comp.net_charge_e,
         "neutral": comp.neutral, "mass (amu)": comp.mass_amu,
         "file charges sum (e)": comp.model_charge_e},), tuple(comp.notes))
    if comp.oxide_mol_percent:
        rows = [{"oxide": oxide, "mol %": value, "atoms left over": None}
                for oxide, value in comp.oxide_mol_percent.items()]
        rows += [{"oxide": f"{anion} not in the oxides", "mol %": None,
                  "atoms left over": value}
                 for anion, value in (comp.anion_unassigned or {}).items()]
        tables["oxide mol %"] = Table("oxide mol %", tuple(rows),
                                      tuple(comp.notes))
    minima_rows = []
    for (a, b), m in result.minima.items():
        if a > b:
            continue
        minima_rows.append({
            "pair": f"{a}-{b}", "first minimum (Å)": m.r_ang,
            "source": m.source, "g at the point (1)": m.g_value,
            "g standard error (1)": m.g_std_error,
            "depth (standard errors)": m.depth_std_errors,
            "uncorrelated level (1)": m.level,
            "floor from (Å)": None if m.floor_r_ang is None
            else m.floor_r_ang[0],
            "floor to (Å)": None if m.floor_r_ang is None
            else m.floor_r_ang[1],
            "method": m.method, "no minimum because": m.reason or None,
            "notes": " | ".join(m.notes) or None})
    tables["g(r) first minima"] = Table("g(r) first minima",
                                        tuple(minima_rows))
    tables["distance cutoffs"] = Table("distance cutoffs", tuple(
        {"pair": f"{c}-{a}", "cutoff (Å)": v,
         "source": result.cutoff_sources[(c, a)]}
        for (c, a), v in result.cutoffs_ang.items()))
    suffix = {"bv": " (BV)", "distance": " (distance)"}
    for section, name, container in result.results():
        tables[f"{name}{suffix.get(section, '')}"] = container
    return AnalysisOutput("glass", tables, result.provenance, raw=result,
                          notes=tuple(result.notes), seconds=seconds)
