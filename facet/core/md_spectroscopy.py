"""Where an MD model meets two experiments: NMR and EXAFS.

The glass descriptors (CN, Q^n, N4, O speciation, angles) are measurements of
the model. What is known about a real glass comes largely from two
spectroscopies that do not see those descriptors directly, and this module
holds the three ways FACET puts a model beside them. None of them fits
anything and none of them judges a model: each lays the model's numbers next
to the measured ones, with the arithmetic and every input stated. Every
result that can be exported (the comparison table, the spectrum, the shells,
the FEFF list, the averaged chi(k)) carries the ``md_stats.Provenance``
header of the trajectory and settings behind it, or a note that the caller
gave none.

1. FRACTIONS BESIDE AN NMR FIT
------------------------------
A Q^n, N4, Al CN or BO/NBO fraction from an NMR deconvolution is compared with
the same fraction counted in the model (:func:`compare_fractions`). The model
side is a :class:`~.md_stats.Distribution` (mean and spread across frames) or
a plain mapping, so this module does not depend on how the glass module
counts; the measured side is a :class:`MeasuredFractions` the user fills from
their own fit, with its uncertainties and its source. The table holds the
model mean, its spread across frames, the measured value, its uncertainty as
entered, and the difference model - measured. The spread across frames and
the measured uncertainty are different kinds of number (the first says how
much the frames differ, the second is the fit's own error estimate), so they
are listed side by side and never combined into a z-score. Keys are matched by
their label ('Q4', 'B4', 'Si-Al'), and a measured key with no model
counterpart is refused rather than left out. NMR reports N4 as B4 / (B3 + B4);
a model whose boron also takes other coordinations at the chosen threshold
has a 'B4' fraction of all boron that differs from that ratio, and
:func:`renormalised` forms the NMR definition from the per-frame counts. When
a complete measured set leaves out a model category of non-zero mean, the two
sets are fractions of different totals, and the comparison says so with the
sum of the model keys the measurement matches.

Every value of a fraction set lies in [0, 1] exactly. Rounded values from a
paper do not sum to 1 exactly, so the user may state the rounding as
``sum_tol``; a ``sum_tol`` wider than rounding can explain is refused: n values
written to a last decimal place q miss 1 by at most n q / 2 in total, q read
from the finest place among the values as entered (0.33 -> 0.01).

2. SHIFTS FROM STRUCTURE-SHIFT CORRELATIONS
-------------------------------------------
Published correlations tie an isotropic chemical shift to a local structural
descriptor: a 29Si shift to the mean Si-O-T angle or to the number of Al
next-nearest neighbours, a 27Al shift to the coordination number, a 23Na
shift to the mean Na-O distance. **FACET ships no coefficients.** A
:class:`Correlation` is filled by the user: the nucleus, the terms with their
coefficients, the literature reference they come from, the range of
descriptor values each term was fitted over, and the shift reference
compound. :func:`predict_shifts` refuses, with 'TODO: need reference', a
correlation that lacks its coefficients or its reference.

The shift of an atom is ``intercept_ppm + sum(coefficient_ppm * x ** power)``
over the terms, x the atom's value of the term's descriptor. Every descriptor
except 'user' is computed from the bonds the CN counts (``bulk.bonds_at``),
so the shift and the CN rest on one bond definition:

* ``'T-O-T angle'``: for a network former, the angles of the bridges it takes
  part in (T-X-T', X the correlation's bridging anion, an anion bonded to two
  or more formers); for the anion element itself, the angles centred on it.
  Each angle is ``atan2(|u x v|, u . v)``, as ``planes.interplanar_angle``
  computes angles, because acos of the dot product loses half its digits near
  0 and 180 degrees. The per-atom value is the mean, over those angles, of one
  of :data:`ANGLE_FUNCTIONS` (the angle in degrees, its cosine,
  cos/(cos - 1), sec, sin of the half angle): published 29Si correlations use
  each of these forms, and which one a correlation uses is stated by its
  reference.
* ``'Qn'``: bridging anions on a former; ``'bridges to'``: bridges from a
  former to formers of one element (the n of Si(nAl)); ``'CN'``: bonds of the
  atom; ``'bonds to'``: bonds to one element; ``'mean bond length'``: mean
  bond length, optionally to one element; ``'BVS'``: bond-valence sum over the
  bonds.
* ``'user'``: per-atom values the caller computed, under a name and a unit.

A term's power is a whole number other than 0; a negative power divides by
the descriptor, and an atom whose value is 0 then has no value of the term.
Two terms that share a label are one descriptor (they differ in power only);
a 'user' term named like a computed descriptor is refused.

The bonds handed to :func:`predict_shifts` are checked to be the frame's:
every bond vector is formed again from the frame's fractions and box with
``bulk._pair_geometry``, as the bulk engine forms it, and a difference above
1e-9 Å is refused (as ``md_order`` and ``md_network`` check theirs). A
descriptor that cannot exist in the frame is refused rather than returned as
zeros: 'bridges to' an element outside the formers, 'bonds to' or 'mean bond
length' to an element on the same side of every bond (two cations), a bridge
descriptor when the frame holds no bridging anion. A partner absent from the
frame gives values of 0 (or none), with a note.

An atom with no value of a descriptor (a Q0 silicon has no Si-O-Si angle, an
atom with no bond has no mean bond length) has no shift: it is counted, noted,
and in no spectrum. An atom whose descriptor lies outside the stated range is
kept, its shift being the correlation extrapolated; it is counted per term
and noted, and its part of the spectrum is also given on its own. A term
whose range was not entered is noted with 'TODO: need reference'.

The spectrum (:func:`broadened_spectrum`, :func:`nmr_spectrum`) places one
line of unit area at each predicted shift, Gaussian or Lorentzian, with the
full width at half maximum the user gives, on the user's ppm grid
(Gaussian: sigma = FWHM / (2 sqrt(2 ln 2)); Lorentzian: half width
gamma = FWHM / 2). It therefore integrates, over all shifts, to the number of
atoms with a shift. Over a finite grid it integrates to the part of each line
inside the grid, which is computed analytically per frame (erf for the
Gaussian, arctan for the Lorentzian) beside the trapezoid integral, so the
discretisation of the grid can be read off; the result states the largest
grid step against the FWHM and the largest relative difference of the two
areas, and says so when a step reaches the FWHM (a line can then fall between
grid points: on a Gaussian of FWHM equal to the step the trapezoid area moved
by up to 5.7 %, measured). Shifts beyond the grid's ends are counted per
frame. The grid has to increase; an axis in NMR plotting order is passed
reversed. **Quadrupolar lineshapes are not
simulated.** For a nucleus of spin I > 1/2 (27Al, 11B, 17O, 23Na) a measured
MAS line carries a second-order quadrupolar shape and shift; here a line
sits at the isotropic chemical shift the correlation predicts.

3. EXAFS FROM THE MODEL
-----------------------
**Absorber-centred partial g(r).** For an absorber element A and each
neighbour element B: the A-B distances from the frame's one pair search,
deposited linearly on the grid of ``pdf.pair_distribution`` with
``pdf._deposit`` (which conserves each pair's weight and keeps the first
moment of a peak to O(dr^2)), normalised as pdf.py normalises,
g_AB(r) = R_AB(r) / (4 pi r^2 rho_B), with R_AB the neighbours per absorber
per Å and rho_B = N_B / V. A pair between the last grid point and one step
past it deposits part of its weight on the last point, so the pairs have to
be searched to the last point plus dr, as ``glass.partial_rdf`` asks; a
search that stopped at r_max left the last point of an Al-O g(r) of a
2 880-atom glass at 0.53 against 1.14 (one frame, r_max 6 Å, dr 0.01 Å,
measured 2026-10-07), and is refused. N_AB(r), the running coordination
number, is an exact count of the distances, not an integral of the
histogram. r_max, and
the outer edge of every shell, is refused beyond half the smallest
perpendicular width of the box, where the model's periodic images start to
correlate with themselves (a shell past it counts a pair once per image:
152 080 of 1.5 million Si-O pairs twice in a 20.6 Å shell of a 35 Å silica
box, measured). :func:`md_exafs` checks this on the first frame before any
pair search.

**Shell limits** come from a measured minimum, never a typed distance: the
first minimum of the frame-averaged g(r) by ``glass.first_minimum``, under
the caller's ``glass.MinimumMethod`` (rule, smoothing, flat rule and a margin
in counting errors, none of them defaulted), with the uncorrelated level
(1, or 1 - 1/N_A for the absorber's own element) and the Poisson counting
errors of g (``glass.pair_counts_per_unit_g`` over the frames averaged)
that the glass analysis passes, so that the two modules give one cutoff for
one g(r); or a limit the user gives. :func:`first_minimum` adds the highest
point of g below the limit, the floor, a note when g at the limit is not
below the level, and a note when g rises beyond the limit above that highest
point (the rule then stopped before the main peak, on a rise at its foot).
A second rule written here once (the first point after
the first maximum where g rises, values as they stand) put the limit on a
noise wiggle of a real model's peak: Al-O of a 2 880-atom aluminosilicate
glass at 1.77 Å with g = 19.6 there, so the 'first shell' held N = 1.79 of
about 4 O (measured 2026-10-07). Through glass, on the same 20-frame SHIK
models (absorber-centred g at 0.01 Å, no smoothing), the 'valley' rule at a
margin of 2 errors gave Al-O 2.37 Å (N 4.049), Al-Si 3.62 Å (N 2.973) and
Si-Si 3.48 Å (N 4.037), against 2.41, 3.61 and 3.47 Å with running CN
4.053, 2.973 and 4.037 from LAMMPS ``compute rdf`` (0.02 Å bins), and Si-O
2.27 Å (N 4.007; LAMMPS 2.03 Å, 4.005, inside the 1.91-2.64 Å floor
reported); the 'first local minimum' rule at margin 0 stopped inside the
Al-O, Al-Si, Al-Al and Si-Si peaks, each time with the note that g at the
limit is not below the level (Al-Si, Al-Al and Si-Si with the second note
too), and before the Al-Na peak, at 2.74 Å with g = 0.84 there and
N = 0.085 (LAMMPS 4.29 Å, 2.709), which only the second note marks; at a
margin of 2 errors, under either rule, the second note was given for no
pair (re-run 2026-10-07 with that note in place, every limit unchanged).

**Cumulants** of the distances in a shell, per frame and pooled
over frames: N (neighbours per absorber), R = C1 (the mean), sigma^2 = C2,
C3 and C4 = mu4 - 3 mu2^2, population cumulants computed in two passes with
``math.fsum``. These are the coefficients of the cumulant expansion of EXAFS
[1, 2]: for one shell the expansion carries
exp(2ik C1 - 2k^2 C2 - (4/3) i k^3 C3 + (2/3) k^4 C4). They are the cumulants
of the distance distribution itself. An EXAFS fit returns those of the
effective distribution, which the 1/r^2 of the EXAFS amplitude and the
photoelectron mean free path also weight; ``weighting='1/r^2'`` applies the
first of the two (the second needs lambda(k) from FEFF), and the result says
which.

**Configurational averaging with FEFF** [3, 4]. K absorber atoms are drawn
across the chosen frames (:func:`sample_absorbers`, seeded, without
replacement), each one's cluster is taken from that frame's pair search and
written as ``feff.inp`` (:func:`write_feff_input`) in the format
``exporters.write_feff`` writes for a crystal, so FEFF reads both alike, and
the list of clusters goes into ``clusters.csv``. FACET does not run FEFF.
An ``out_dir`` that already holds a ``clusters.csv`` or a drawn ``feff.inp``
is refused unless ``overwrite=True``; on overwrite the FEFF output beside each
rewritten ``feff.inp`` is removed (it belongs to the input replaced), and
cluster directories of an earlier draw that the new list omits are noted.
The header of each ``feff.inp`` names the FACET version, the source file,
the frame, its timestep and the absorber's id and row, and no bond-valence
setting (the cluster is cut by distance alone). FEFF8L stops on more than 10
unique potentials (below), so a cluster that needs more carries a comment
and a note.

When FEFF has run in those directories, :func:`average_feff_chi` reads each
one back with ``exafs.read_feff_directory``, sums a stated set of its paths
with ``exafs.chi_from_paths`` at the user's S0^2 on one k grid, and averages
chi(k) over the clusters, with the spread across clusters. The path set is
by default the one FEFF's own ff2x summed into ``chi.dat``, read from that
file's header: FEFF8L writes a ``feffNNNN.dat`` for every path of
curved-wave amplitude ratio 2/3 of its criterion or more, and ff2x keeps
those of the criterion itself (4 % by default), so summing every path file
differs from FEFF's chi(k): by 8.1 % of max|chi| over k = 3-14 1/Å for four
Al K clusters of 6 Å in a sodium aluminosilicate glass (18-26 of the 132-164
path files of a cluster in that band, measured 2026-10-07). Every path file or the paths
above a stated amplitude ratio are the other two choices. A directory is
left out, with the reason, when its FEFF output is incomplete (a path file
that ``files.dat`` lists is absent or unreadable, or a path file is not in
the list), stale (``files.dat`` older than ``feff.inp``, or its Rmax not the
RPATH of ``feff.inp``), or does not cover the k grid; a directory with no
``feff.inp`` beside its output is used and noted as unchecked, and one given
twice is refused. The number of paths
summed is stated per cluster. Every path is at sigma^2 = 0: the clusters are
static snapshots, and the disorder of the model is in their spread.

ONE PAIR SEARCH PER FRAME
-------------------------
Nothing here searches for neighbours. The NMR descriptors take ``bulk.Bonds``;
the EXAFS functions take the frame's ``bulk.PairTable`` (or the blocks of
``bulk.iter_pairs``), keep the pairs centred on the absorber, and check with
``bulk``'s own coverage test that the pairs are this frame's and cover every
atom once. The trajectory-level helpers (:func:`md_nmr`, :func:`md_exafs`,
:func:`export_feff_inputs`) call ``bulk.iter_pairs`` once per frame, each
to the radius that frame needs: :func:`md_exafs` searches to the FEFF
cluster radius only in a frame that holds a drawn cluster (before, one
12 Å cluster made each of 20 frames search to 12 Å). The one exception is
stated where it happens: when the shell limits of
:func:`md_exafs` come from the frame-averaged g(r), they are known only after
every frame has been read, so a second pass reads each frame again and
searches only to the outermost shell limit (as the glass analysis does for its
distance-cut descriptors), and the first pass only as far as the g(r) grid
needs; ``MdExafs.searches_per_frame`` says 1 or 2.

FEFF, CHECKED OUTSIDE THE SUITE
------------------------------
The test suite never runs FEFF. By hand, with the FEFF8L that ships with
xraylarch on this machine ('Feff8L (EXAFS) release 0.1'):

* 2026-10-06: three 5 Å clusters of a quartz 2x2x2 supercell frame written
  by :func:`export_feff_inputs` (Si K edge, RPATH 4 Å). FEFF accepted each
  input as written and wrote 24 paths; every single-scattering r_eff in its
  ``files.dat`` lay within 3e-5 Å of a cluster distance (FEFF prints four
  decimals).
* 2026-10-07, an independent verification: 15 clusters of two SHIK glass
  models (4 Al K and 4 Na K clusters of 6 Å from a 2 880-atom sodium
  aluminosilicate; 4 Si K clusters of 6 Å, 2 of 8 Å and 1 of 12 Å from a
  3 000-atom silica). FEFF accepted every input; the ATOMS blocks matched a
  minimum-image cluster built by separate code from the raw LAMMPS dump to
  5e-6 Å; every atom within RPATH had its single-scattering path, with r_eff
  within 5.04e-5 Å of its distance. FEFF printed 'WARNING: rmax > distance
  to most distant atom. Some paths may be missing. rmax, ratx 5.0 0.0' for
  all 15 inputs (counted in its logs): the path coverage just stated shows
  that no path was missing, and ratx = 0 points to the way FEFF8L reports
  it, not to the input.
* :func:`average_feff_chi` at S0^2 = 1 against FEFF's own ``chi.dat`` on
  those runs (re-measured 2026-10-07), the largest |difference| per cluster
  as a fraction of the largest |chi.dat| in the same k range: with the
  default path set (the paths ff2x summed) 0.42 % (Al), 0.94 % (Na), 1.95 %
  (Si, 6 Å), 1.60 % (Si, 8 Å), 2.24 % (Si, 12 Å) and 0.90 % (quartz) over
  k = 3-14 1/Å, at most 0.20 % over k = 5-14 and up to 3.6 % over
  k = 2.5-3.5. That residual is the linear interpolation of the path files,
  which FEFF8L writes on a coarse k grid (steps of 0.1 1/Å to k = 2, 0.2 to
  6, 0.5 to 10, 1.0 to 20), where ff2x works from its own finer data; no
  phase step above pi was found in the 2 826 path files. Summing every path
  file instead differed by 8.1 % (Al, mean of 4 clusters) and up to 12.6 %
  (one Si cluster) over k = 3-14.
* 2026-10-07: with 11 POTENTIALS entries (ipot 0-10) FEFF8L's genfmt
  stopped, with 13 FEFF8L itself, and its batch file returned 0 both times
  (``_FEFF8L_MAX_POTENTIALS``).

TIMINGS
-------
Measured 2026-10-06 on this machine (Windows 11, i5-13420H, Python 3.11.9,
numpy 2.4.6, scipy 1.15.1), pinned to the P-cores, on a machine shared with
other sessions; median of 5 runs after one untimed run (3 for the
trajectory helpers). The frame is ``tools/bench_md.make_box(22)``: 10 648
atoms (2 701 Si, 6 379 O, 1 568 Na) on a jittered 2.3 Å grid in a 50.6 Å
cube, chemistry meaningless, 15 222 bonds at the default v_bond. For scale,
``bulk.analyse_frame`` took 0.80 s on it.

* :func:`predict_shifts`, 29Si with T-O-T angle + Qn + bridges to Si
  (2 701 Si): 8.9 ms; 23Na with CN + mean bond length: 1.9 ms. With the
  check that the bonds are the frame's (re-measured 2026-10-07, a run in
  which ``bulk.analyse_frame`` took 0.67 s): 11.1 ms and 6.4 ms, of which
  the check is 2.9 ms.
* :func:`broadened_spectrum`, 2 586 lines on 4 001 points: Gaussian 0.14 s
  with a 6 MB peak, Lorentzian 0.07 s. Unpinned, the Gaussian took
  0.27-0.70 s, against 0.47-0.83 s and a 197 MB peak before the 40-sigma
  window and the in-place chunks. :func:`line_area_in_grid` under 1 ms.
  :func:`nmr_spectrum` for one frame: 0.15 s (0.11 s on 2026-10-07, with
  the grid and beyond-grid bookkeeping).
* :func:`md_nmr`: 0.97 s a frame, of which ``bulk.analyse_frame`` is 0.80 s.
* :func:`absorber_pairs` (Si) from a ready PairTable: 49 ms at 6 Å
  (851 840 pairs) and 0.18 s at 10 Å (3.7 million); with the
  ``bulk.iter_pairs`` search included, 0.61 s and 2.4 s.
* :func:`absorber_rdf`, 3 neighbour elements at 0.01 Å: 49 ms to 6 Å and
  0.21 s to 10 Å (942 872 Si-centred pairs, 54 MB peak).
  :func:`first_minimum` (through glass, valley rule, sigma 0.02 Å, margin
  2, 600 points): 0.29 ms (2026-10-07). :func:`shell_cumulants` (Si-O first
  shell): 41 ms.
* :func:`feff_cluster` + :func:`write_feff_input`: 2.2 ms a cluster of 6 Å.
  :func:`average_feff_chi`: 15 ms a FEFF run of 24 paths; with the checks
  of completeness and staleness, 0.26 s for 4 runs of 132-164 path files
  (2026-10-07).
* :func:`md_exafs` with the limits from the averaged g(r) (two passes):
  0.79 s a frame to r_max = 6 Å and 2.7 s to 10 Å, nearly all of it the
  pair search of ``bulk``.

REFERENCES
----------
[1] G. Bunker, "Application of the ratio method of EXAFS analysis to
    disordered systems", *Nuclear Instruments and Methods in Physics
    Research* 207 (1983) 437-444, https://doi.org/10.1016/0167-5087(83)90655-5
[2] J. J. Rehr and R. C. Albers, "Theoretical approaches to x-ray absorption
    fine structure", *Reviews of Modern Physics* 72 (2000) 621-654,
    https://doi.org/10.1103/RevModPhys.72.621
[3] A. L. Ankudinov, B. Ravel, J. J. Rehr and S. D. Conradson, "Real-space
    multiple-scattering calculation and interpretation of x-ray-absorption
    near-edge structure", *Physical Review B* 58 (1998) 7565-7576,
    https://doi.org/10.1103/PhysRevB.58.7565
[4] A. Kuzmin and J. Chaboy, "EXAFS and XANES analysis of oxides at the
    nanoscale", *IUCrJ* 1 (2014) 571-589,
    https://doi.org/10.1107/S2052252514021101
"""
from __future__ import annotations

import dataclasses
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import numpy as np

from . import bv, exafs, glass
from . import md_stats
from .bulk import (Bonds, PairTable, _blocks_of, _check_frame, _Coverage,
                   _pair_geometry, analyse_frame, bonds_at, iter_pairs)
from .md_model import Frame, ModelOxidation, validate_symbol
from .md_stats import (FACET_VERSION, Distribution, Histogram, Provenance,
                       Scalar, Series, check_fractions, mean_and_spread)
from .pdf import _deposit

__all__ = [
    "TODO_REFERENCE", "LINESHAPES", "ANGLE_FUNCTIONS", "DESCRIPTORS",
    "WEIGHTINGS", "PATH_SETS", "SPECTRUM_CHUNK_ATOMS",
    "QUADRUPOLAR_NOTE", "AnalysisCancelled",
    # NMR fractions
    "MeasuredFractions", "FractionComparison", "compare_fractions",
    "comparison_rows", "renormalised",
    # NMR shifts
    "Term", "Correlation", "ShiftPrediction", "predict_shifts",
    "broadened_spectrum", "line_area_in_grid", "NmrSpectrum", "nmr_spectrum",
    "shift_histogram", "md_nmr",
    # EXAFS
    "AbsorberPairs", "absorber_pairs", "AbsorberRDF", "absorber_rdf",
    "ShellLimit", "first_minimum", "distance_cumulants", "ShellCumulants",
    "shell_cumulants", "AveragedShell", "average_shells", "FeffCluster",
    "feff_cluster", "write_feff_input", "sample_absorbers", "FeffRequest",
    "FeffClusterRecord", "FeffExport", "export_feff_inputs", "AveragedChi",
    "average_feff_chi", "MdExafs", "md_exafs",
]

# ---------------------------------------------------------------------------
# constants: each one is a choice or a label, and says so
# ---------------------------------------------------------------------------

# The words every refusal for a missing literature value carries.
TODO_REFERENCE = "TODO: need reference"

LINESHAPES = ("gaussian", "lorentzian")

# The functions of a bridge angle a 'T-O-T angle' term can take. Labels, not
# parameters: 'deg' is the angle in degrees, 'sin(half)' is sin(angle / 2).
ANGLE_FUNCTIONS = ("deg", "cos", "cos/(cos-1)", "sec", "sin(half)")

DESCRIPTORS = ("T-O-T angle", "Qn", "bridges to", "CN", "bonds to",
               "mean bond length", "BVS", "user")

WEIGHTINGS = ("none", "1/r^2")

# Which FEFF paths average_feff_chi sums in a directory: those FEFF's own ff2x
# summed into chi.dat (listed in its header), or every path file FEFF wrote.
# A number in place of either keeps the paths whose curved-wave amplitude
# ratio in files.dat is at least that many per cent.
PATH_SETS = ("chi.dat", "all")

# How far a bond vector handed to predict_shifts may sit from the vector the
# frame's fractions and box give. A numerical tolerance, as
# md_network.VECTOR_TOL_ANG: both are formed by bulk._pair_geometry and agree
# to the bit on the frame the bonds came from; the bonds of frame 0 laid on
# frame 19 of a 2 880-atom glass differed by 34.7 Å (measured).
_VECTOR_TOL_ANG = 1e-9

# How far a search may fall short of the g(r) grid's last point plus dr. A
# numerical tolerance: numpy.arange's last point can exceed r_max by rounding
# (3.0000000000000004 for steps of 0.1 to 3), so a search to r_max + dr falls
# short of it by about 5e-16 Å, and a pair inside that gap would deposit a
# weight below 1e-9 / dr on the last point.
_GRID_ROUNDING_ANG = 1e-9

# The most unique potentials (POTENTIALS entries, ipot 0 .. 9) with which
# the FEFF8L shipped with xraylarch ran on this machine (2026-10-07, the
# elements of a real 6 Å glass cluster relabelled): with 11 entries its
# genfmt stopped ('fatal error reading feff.pad'), with 13 FEFF stopped
# ('Unique potentials must be between 0 and 11'), and its batch file
# returned 0 both times. A measured property of one FEFF build, used for a
# comment and a note only.
_FEFF8L_MAX_POTENTIALS = 10

# What FEFF8L (xraylarch's) wrote beside feff.inp in a cluster directory on
# this machine. On overwrite=True these are removed with the input they
# belong to, so a stale chi(k) cannot be read back as the new one's.
_FEFF_OUTPUTS = ("files.dat", "paths.dat", "list.dat", "chi.dat", "xmu.dat",
                 "feff[0-9]*.dat", "log*.dat", "*.pad", "atoms.json",
                 "ff2x.json", "genfmt.json", "geom.json", "global.json",
                 "libpotph.json", "path.json", "pot.json", "xsect.json",
                 "xsph.json")

# FEFF prints Rmax in files.dat with three decimals; an RPATH that rounds to
# it within half of that last place is the same RPATH. A parsing tolerance.
_RMAX_PRINT_TOL_ANG = 5e-4

# How many atoms' lines are evaluated on the grid at once. A speed and memory
# choice: 128 atoms on a 4 001-point grid is a 4 MB float64 buffer, and on
# 2 586 lines x 4 001 points chunks of 64-128 ran 0.50 s against 0.58-0.71 s
# for 256-2 048 (measured, unpinned, before the Gaussian window below).
SPECTRUM_CHUNK_ATOMS = 128

# FWHM = 2 sqrt(2 ln 2) sigma for a Gaussian: arithmetic, not a parameter.
_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))

# A Gaussian line is evaluated within this many sigma of its centre only.
# Not an approximation: beyond 38.6 sigma exp(-x^2 / 2) is below the smallest
# float64 (4.9e-324) and evaluates to exactly 0.0, so at 40 sigma the window
# leaves out nothing but zeros.
_GAUSSIAN_ZERO_SIGMAS = 40.0

QUADRUPOLAR_NOTE = (
    "quadrupolar lineshapes are not simulated: each atom contributes one line "
    "of the chosen shape and width at its predicted isotropic chemical shift. "
    "For a nucleus of spin I > 1/2 (27Al, 11B, 17O, 23Na) a measured MAS line "
    "also carries a second-order quadrupolar shape and shift, which are not "
    "in this spectrum")

_EDGE = re.compile(r"^(K|L[1-3]|M[1-5]|N[1-7])$")
_NUCLEUS = re.compile(r"^(\d{1,3})([A-Za-z]{1,2})$")

# chi.dat's header, as FEFF8L's ff2x writes it: one '#' line per path summed
# (index, sig2, curved-wave amplitude ratio, degeneracy, legs, r_eff), then
# '#   146/ 164 paths used', and the filter it applied.
_CHI_PATH = re.compile(r"^#\s*(\d+)\s+-?\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+"
                       r"\s+\d+\.\d+(?:\s+-?\d+\.\d+)?\s*$")
_CHI_USED = re.compile(r"^#\s*(\d+)\s*/\s*(\d+)\s+paths used")
_CHI_FILTER = re.compile(r"amplitude ratio filter\s+(\d+(?:\.\d*)?)\s*%",
                         re.IGNORECASE)
_RMAX = re.compile(r"Rmax\s*=\s*(\d+(?:\.\d*)?)")


class AnalysisCancelled(Exception):
    """Raised by the trajectory-level helpers when ``cancelled()`` returns True."""


# ---------------------------------------------------------------------------
# small checks
# ---------------------------------------------------------------------------

def _text(value, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} needs a non-empty text, not {value!r}")
    return value.strip()


def _number(value, what: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{what} is {value!r}, not a number")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{what} is {value!r}, not a number") from None
    if not math.isfinite(out):
        raise ValueError(f"{what} is {value!r}; a finite number is needed")
    return out


def _positive(value, what: str) -> float:
    out = _number(value, what)
    if out <= 0.0:
        raise ValueError(f"{what} is {value!r}; a number above 0 is needed")
    return out


def _whole(value, what: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or \
            not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{what} is {value!r}; a whole number of {minimum} "
                         "or more is needed")
    return int(value)


def _label(key) -> str:
    return md_stats._key_label(key)


def _decimal_place(value: float) -> Decimal:
    """The last decimal place of a number as Python writes it (0.33 -> 0.01)."""
    return Decimal(1).scaleb(Decimal(repr(float(value))).as_tuple().exponent)


def _check_set(values: Sequence[float], what: str, tol: float,
               complete: bool) -> None:
    """A fraction set: each value in [0, 1] exactly, the sum within ``tol``.

    ``tol`` above md_stats.FRACTION_TOL is a stated rounding of the values,
    so it is refused when wider than rounding can explain: n values written
    to a last decimal place q miss 1 by at most n q / 2, q taken from the
    finest place among the values as entered. Rounding cannot take a
    fraction out of [0, 1], so ``tol`` never widens that interval.
    """
    for value in values:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{what}: a fraction is {value!r}, outside [0, 1]; rounding a "
                "fraction cannot take it out of that interval, so sum_tol "
                "does not widen it")
    if tol > md_stats.FRACTION_TOL:
        place = min(_decimal_place(v) for v in values)
        bound = len(values) * place / 2
        if Decimal(repr(tol)) > bound:
            raise ValueError(
                f"{what}: sum_tol {tol!r} is wider than rounding can explain: "
                f"{len(values)} values written to {place} at the finest miss "
                f"1 by at most {len(values)} x {place / 2} = {bound}")
    total = math.fsum(values)
    if complete and abs(total - 1.0) > tol:
        raise ValueError(f"{what}: the fractions sum to {total!r} (|sum - 1| "
                         f"= {abs(total - 1.0)!r}; tolerance {tol!r})")
    if not complete and total - 1.0 > tol:
        raise ValueError(f"{what}: the partial set sums to {total!r}, above 1 "
                         f"(sum - 1 = {total - 1.0!r}; tolerance {tol!r})")


def _by_label(keys, what: str) -> dict[str, object]:
    """label -> key; two keys with one label are refused (4 and '4')."""
    out: dict[str, object] = {}
    for key in keys:
        label = _label(key)
        if label in out:
            raise ValueError(f"{what}: the keys {out[label]!r} and {key!r} "
                             f"share the label {label!r}")
        out[label] = key
    return out


def _former_set(formers, anion: str | None) -> frozenset[str] | None:
    if formers is None:
        return None
    if isinstance(formers, (str, bytes)):
        raise ValueError("formers needs a collection of element symbols, not "
                         "one string")
    out = frozenset(validate_symbol(str(s)) for s in formers)
    if anion is not None and anion in out:
        raise ValueError(f"{anion} is the bridging anion and cannot also be "
                         "a network former")
    return out


def _frame_indices(trajectory, frames) -> list[int]:
    if frames is None:
        return list(range(trajectory.n_frames))
    if isinstance(frames, (str, bytes)):
        raise ValueError("frames needs a sequence of frame indices")
    out = []
    for k in frames:
        k = _whole(k, "a frame index", 0)
        if k >= trajectory.n_frames:
            raise ValueError(f"frame {k} is outside the {trajectory.n_frames} "
                             "readable frames")
        out.append(k)
    if not out:
        raise ValueError("no frame was chosen")
    if len(set(out)) != len(out):
        raise ValueError("a frame index occurs more than once")
    return sorted(out)


def _tick(progress, cancelled, done: int, total: int) -> None:
    if cancelled is not None and cancelled():
        raise AnalysisCancelled(f"cancelled after {done} of {total} frame "
                                "passes")
    if progress is not None:
        progress(done, total)


# ===========================================================================
# 1. fractions beside an NMR fit
# ===========================================================================

@dataclass(frozen=True, eq=False)
class MeasuredFractions:
    """Fractions from the user's own NMR fit, with their uncertainties.

    ``descriptor`` names the set ('Si Qn', 'B N4', 'Al CN', 'O speciation');
    ``values`` maps a key ('Q4', 4, 'B4', 'NBO') to a fraction in [0, 1];
    ``uncertainties`` maps some or all of those keys to an absolute
    uncertainty in the same units (None or a missing key: not given);
    ``source`` says where the numbers come from (sample, nucleus, fit) and is
    required.

    ``complete=True`` (the set lists every category): the values sum to 1
    within ``sum_tol``, whose default is ``md_stats.FRACTION_TOL``.
    Fractions copied from a paper are rounded, so 0.33 + 0.33 + 0.33 is
    refused at that default and accepted at ``sum_tol=0.015``: three values
    written to 0.01 miss 1 by at most 3 x 0.005, which the user states. A
    ``sum_tol`` wider than that bound (n values, half their finest decimal
    place each) is refused, since rounding cannot explain it.
    ``complete=False`` (a partial set, N4 alone): the sum is at most 1 within
    ``sum_tol``. In both cases every value lies in [0, 1] exactly.
    """

    descriptor: str
    values: Mapping
    source: str
    uncertainties: Mapping | None = None
    complete: bool = True
    sum_tol: float = md_stats.FRACTION_TOL

    def __post_init__(self) -> None:
        put = object.__setattr__
        descriptor = _text(self.descriptor, "descriptor")
        source = _text(self.source, f"{descriptor}: source (sample, nucleus, "
                                    "fit)")
        if not isinstance(self.values, Mapping) or not self.values:
            raise ValueError(f"{descriptor}: values needs a non-empty mapping "
                             "of key to measured fraction")
        values = {}
        for key, value in self.values.items():
            k = md_stats._key(key, descriptor)
            values[k] = _number(value, f"{descriptor}: the measured fraction "
                                       f"of {_label(k)}")
        _by_label(values, f"{descriptor} (measured)")
        uncertainties = {}
        if self.uncertainties is not None:
            if not isinstance(self.uncertainties, Mapping):
                raise ValueError(f"{descriptor}: uncertainties needs a mapping "
                                 "of key to uncertainty, or None")
            for key, value in self.uncertainties.items():
                k = md_stats._key(key, descriptor)
                if k not in values:
                    raise ValueError(f"{descriptor}: an uncertainty is given "
                                     f"for {_label(k)}, which has no measured "
                                     "value")
                u = _number(value, f"{descriptor}: the uncertainty of "
                                   f"{_label(k)}")
                if u < 0.0:
                    raise ValueError(f"{descriptor}: the uncertainty of "
                                     f"{_label(k)} is {u!r}, below 0")
                uncertainties[k] = u
        if not isinstance(self.complete, (bool, np.bool_)):
            raise ValueError(f"{descriptor}: complete is True or False")
        tol = md_stats._tolerance(self.sum_tol)
        what = f"{descriptor} (measured, {source})"
        _check_set(list(values.values()), what, tol, bool(self.complete))
        if self.complete:
            check_fractions(values, what, tol=tol)
        put(self, "descriptor", descriptor)
        put(self, "source", source)
        put(self, "values", values)
        put(self, "uncertainties", uncertainties)
        put(self, "complete", bool(self.complete))
        put(self, "sum_tol", tol)


@dataclass(frozen=True, eq=False)
class FractionComparison:
    """One model fraction set beside one measured set, key by key.

    Arrays run over ``keys`` in the model's order. ``measured`` and
    ``uncertainty`` are NaN where the measurement gives no value;
    ``model_std`` is NaN where the model gives no spread (a single frame or a
    plain mapping); ``model_frames`` is None when unknown. ``difference`` is
    model mean - measured, in fraction units. ``provenance`` is the header
    of the analysis behind the model values when the caller gave it.
    """

    descriptor: str
    model_name: str
    keys: tuple
    model_mean: np.ndarray
    model_std: np.ndarray
    model_frames: int | None
    measured: np.ndarray
    uncertainty: np.ndarray
    difference: np.ndarray
    source: str
    notes: tuple[str, ...]
    provenance: Provenance | None = None

    def as_sheets(self) -> dict[str, list[dict]]:
        """Tables for ``exporters.write_xlsx``: provenance, rows, notes."""
        sheets = {}
        if self.provenance is not None:
            sheets["provenance"] = self.provenance.as_rows()
        sheets["comparison"] = self.as_rows()
        sheets["notes"] = [{"note": note} for note in self.notes]
        return sheets

    def as_rows(self) -> list[dict]:
        rows = []
        for c, key in enumerate(self.keys):
            rows.append({
                "descriptor": self.descriptor,
                "key": _label(key),
                "model": self.model_name,
                "model mean": float(self.model_mean[c]),
                "model std (across frames)": float(self.model_std[c]),
                "model frames used": self.model_frames,
                "measured": float(self.measured[c]),
                "measured uncertainty": float(self.uncertainty[c]),
                "model - measured": float(self.difference[c]),
                "measured source": self.source})
        return rows


def _model_fractions(model, model_name, model_sum_tol):
    """(name, keys, mean, std, frames, notes, spread) from a Distribution or
    a mapping; spread is 'frames', 'given' or 'none'."""
    tol = md_stats._tolerance(model_sum_tol)
    if isinstance(model, Distribution):
        if tol != md_stats.FRACTION_TOL:
            raise ValueError("model_sum_tol states the rounding of model "
                             "fractions given as a mapping; a Distribution's "
                             "fractions are counted and take none")
        dist = model.to_fractions() if model.kind == "count" else model
        return (model_name or dist.name, tuple(dist.keys),
                np.array(dist.mean, dtype=np.float64),
                np.array(dist.std, dtype=np.float64), dist.n_frames_used,
                list(dist.notes), "frames")
    if not isinstance(model, Mapping) or not model:
        raise ValueError("the model side needs an md_stats.Distribution or a "
                         "non-empty mapping of key to fraction (or to "
                         "(mean, std))")
    name = model_name or "model"
    keys, means, stds = [], [], []
    pairs = [isinstance(v, (tuple, list)) for v in model.values()]
    if any(pairs) and not all(pairs):
        raise ValueError(f"{name}: give every key a fraction, or every key a "
                         "(mean, std) pair, not a mixture")
    for key, value in model.items():
        k = md_stats._key(key, name)
        if pairs[0]:
            if len(value) != 2:
                raise ValueError(f"{name}: {_label(k)} needs (mean, std)")
            mean = _number(value[0], f"{name}: the mean of {_label(k)}")
            std = _number(value[1], f"{name}: the std of {_label(k)}")
            if std < 0.0:
                raise ValueError(f"{name}: the std of {_label(k)} is below 0")
        else:
            mean, std = _number(value, f"{name}: the fraction of "
                                       f"{_label(k)}"), float("nan")
        keys.append(k)
        means.append(mean)
        stds.append(std)
    what = f"{name} (model)"
    _check_set(means, what, tol, True)
    check_fractions(means, what, tol=tol)
    notes = []
    if tol != md_stats.FRACTION_TOL:
        notes.append(f"{name}: the model fractions sum to "
                     f"{math.fsum(means)!r}, accepted within the stated "
                     f"rounding {tol:g}")
    return (name, tuple(keys), np.array(means), np.array(stds), None, notes,
            "given" if pairs[0] else "none")


def compare_fractions(model, measured: MeasuredFractions, *,
                      model_name: str | None = None,
                      model_sum_tol: float = md_stats.FRACTION_TOL,
                      provenance: Provenance | None = None
                      ) -> FractionComparison:
    """The model's fractions beside the measured ones, with the difference.

    ``model``: an ``md_stats.Distribution`` (fractions, or counts turned into
    fractions), or a mapping key -> fraction, or key -> (mean, std), whose
    fractions sum to 1 (within ``model_sum_tol``, a stated rounding of a
    mapping's values, bounded as ``MeasuredFractions.sum_tol`` is). Keys are
    matched by label; a measured key matching no model key raises ValueError
    naming the model's keys. A model key the measurement does not list keeps
    an empty measured value, never a 0; when the measured set is complete and
    such a key has a non-zero model mean, the two sets are fractions of
    different totals, and a note gives the sum of the matched model keys.
    ``provenance`` (the header of the analysis behind the model values) is
    attached for the export; its absence is noted. Nothing is judged and the
    two kinds of spread are not combined.
    """
    if not isinstance(measured, MeasuredFractions):
        raise ValueError("measured needs a MeasuredFractions")
    if provenance is not None and not isinstance(provenance, Provenance):
        raise ValueError("provenance needs an md_stats.Provenance or None")
    name, keys, mean, std, frames, notes, spread = _model_fractions(
        model, model_name, model_sum_tol)
    model_labels = _by_label(keys, name)
    measured_labels = _by_label(measured.values, measured.descriptor)
    unmatched = [label for label in measured_labels if label not in
                 model_labels]
    if unmatched:
        raise ValueError(
            f"{measured.descriptor}: the measured key(s) "
            f"{', '.join(unmatched)} match no key of the model {name!r} "
            f"({', '.join(model_labels)}); keys are matched by their label")
    n = len(keys)
    values = np.full(n, np.nan)
    uncertainty = np.full(n, np.nan)
    not_given = []
    for c, key in enumerate(keys):
        label = _label(key)
        if label in measured_labels:
            k = measured_labels[label]
            values[c] = measured.values[k]
            if k in measured.uncertainties:
                uncertainty[c] = measured.uncertainties[k]
        else:
            not_given.append(label)
    difference = mean - values
    notes = list(notes)
    notes.append(
        f"{measured.descriptor}: model - measured, in fraction units, key by "
        f"key; the measured values are from {measured.source}")
    if spread == "frames":
        notes.append(
            "the model std is the spread of the per-frame fractions across "
            "frames (ddof = 1), not a standard error; the measured "
            "uncertainty is as entered. The two are listed side by side and "
            "not combined")
    elif spread == "given":
        notes.append(
            f"{name}: the model std is as given by the caller; the measured "
            "uncertainty is as entered. The two are listed side by side and "
            "not combined")
    else:
        notes.append(f"{name}: no model spread was given (model std NaN); "
                     "the measured uncertainty is as entered")
    if not_given:
        if measured.complete:
            notes.append(f"{measured.descriptor}: the complete measured set "
                         f"lists no value for {', '.join(not_given)}; listed "
                         "with an empty measured value, not as 0")
        else:
            notes.append(f"{measured.descriptor}: the measurement gives no "
                         f"value for {', '.join(not_given)}; listed with an "
                         "empty measured value, not as 0")
        others = [c for c, key in enumerate(keys)
                  if _label(key) in not_given and np.isfinite(mean[c])
                  and mean[c] != 0.0]
        if measured.complete and others:
            matched = math.fsum(float(mean[c]) for c, key in enumerate(keys)
                                if _label(key) in measured_labels
                                and np.isfinite(mean[c]))
            notes.append(
                f"{measured.descriptor}: the measured set is complete over "
                f"{', '.join(measured_labels)} and sums to "
                f"{math.fsum(measured.values.values()):.6g}; the model keys it "
                f"matches sum to {matched:.6g}, and the model's "
                + ", ".join(f"{_label(keys[c])} (mean fraction "
                            f"{float(mean[c]):.6g})" for c in others)
                + " hold the rest. The two sets are fractions of different "
                "totals, so each difference mixes the two normalisations; "
                "md_spectroscopy.renormalised(model, ["
                + ", ".join(repr(label) for label in measured_labels)
                + "]) forms a model Distribution over the measured "
                "categories only")
    if not measured.complete:
        notes.append(f"{measured.descriptor}: a partial measured set "
                     "(complete=False), not required to sum to 1")
    elif measured.sum_tol != md_stats.FRACTION_TOL:
        notes.append(f"{measured.descriptor}: the measured set sums to "
                     f"{math.fsum(measured.values.values())!r}, accepted "
                     f"within the stated rounding {measured.sum_tol:g}")
    missing_u = [label for label, k in measured_labels.items()
                 if k not in measured.uncertainties]
    if missing_u:
        notes.append(f"{measured.descriptor}: no uncertainty given for "
                     f"{', '.join(missing_u)}")
    if frames is None:
        notes.append(f"{name}: the model values were given as a mapping, so "
                     "the number of frames behind them is not known here")
    if provenance is None:
        notes.append(f"{name}: no provenance was given, so the trajectory, "
                     "frames and settings behind the model values are not "
                     "recorded with this table (pass the analysis result's "
                     "Provenance as provenance=)")
    return FractionComparison(
        descriptor=measured.descriptor, model_name=name, keys=keys,
        model_mean=mean, model_std=std, model_frames=frames,
        measured=values, uncertainty=uncertainty, difference=difference,
        source=measured.source, notes=tuple(notes), provenance=provenance)


def comparison_rows(comparisons: Sequence[FractionComparison]) -> list[dict]:
    """Every comparison's rows in one table, for ``exporters.write_csv``."""
    rows: list[dict] = []
    for item in comparisons:
        if not isinstance(item, FractionComparison):
            raise ValueError("comparison_rows takes FractionComparison items")
        rows.extend(item.as_rows())
    return rows


def renormalised(model: Distribution, keys: Sequence, *,
                 name: str | None = None) -> Distribution:
    """The fractions of ``keys`` among those keys only, frame by frame.

    NMR gives N4 = B4 / (B3 + B4). From a model distribution over 'B3', 'B4'
    and 'other', ``renormalised(dist, ['B3', 'B4'])`` forms exactly that per
    frame from the counts, then averages over frames; the keys left out of
    the denominator are named in a note with their mean fraction. ``keys``
    are matched by label. A fraction distribution is turned back into counts
    with its ``n_items``; a value that is not a whole number within 1e-6 is
    refused.
    """
    if not isinstance(model, Distribution):
        raise ValueError("renormalised needs an md_stats.Distribution")
    labels = _by_label(model.keys, model.name)
    if isinstance(keys, (str, bytes)):
        raise ValueError("keys needs a sequence of keys, not one string")
    chosen = []
    for key in keys:
        label = _label(md_stats._key(key, model.name))
        if label not in labels:
            raise ValueError(f"{model.name}: no key {label!r} (keys: "
                             f"{', '.join(labels)})")
        if labels[label] in chosen:
            raise ValueError(f"{model.name}: {label!r} is given twice")
        chosen.append(labels[label])
    if not chosen:
        raise ValueError("renormalised needs at least one key")
    if model.kind == "count":
        counts = np.array(model.per_frame, dtype=np.float64)
    else:
        with np.errstate(invalid="ignore"):
            counts = model.per_frame * model.n_items[:, None]
        counts = np.where(np.isnan(counts), 0.0, counts)
    whole = np.rint(counts)
    if np.abs(counts - whole).max() > 1e-6:
        raise ValueError(f"{model.name}: the fractions times the items per "
                         "frame are not whole numbers, so the counts cannot "
                         "be recovered")
    columns = [model.keys.index(k) for k in chosen]
    rows = [{k: int(whole[r, c]) for k, c in zip(chosen, columns)}
            for r in range(whole.shape[0])]
    left = [k for k in model.keys if k not in chosen]
    notes = list(model.notes)
    if left:
        fractions = model.to_fractions() if model.kind == "count" else model
        notes.append(
            f"renormalised over {', '.join(_label(k) for k in chosen)} only: "
            + ", ".join(f"{_label(k)} (mean fraction "
                        f"{float(fractions.mean[model.keys.index(k)]):.6g} of "
                        "all items)" for k in left)
            + " left out of the denominator")
    return Distribution.from_counts(
        rows, name=name or f"{model.name} over "
                           f"{', '.join(_label(k) for k in chosen)}",
        kind="fraction", keys=chosen, frames=model.frames,
        row_kind=model.row_kind, notes=notes)


# ===========================================================================
# 2. shifts from structure-shift correlations
# ===========================================================================

@dataclass(frozen=True)
class Term:
    """One term of a correlation: ``coefficient_ppm * x ** power``.

    ``descriptor`` is one of :data:`DESCRIPTORS`; x is the atom's value of it.
    ``coefficient_ppm`` is in ppm per (descriptor unit)^power, None until the
    user enters it. ``power`` is a whole number other than 0; a negative one
    divides by x, and an atom with x = 0 then has no value of the term
    (counted with the undefined ones). ``valid_range`` is (low, high) in the descriptor's unit
    (after ``angle_function``), the range the correlation was fitted over;
    None means not given. ``angle_function`` (one of
    :data:`ANGLE_FUNCTIONS`) is required for 'T-O-T angle' and refused
    otherwise. ``partner`` (an element) is required for 'bridges to' and
    'bonds to', optional for 'mean bond length', refused otherwise. ``name``
    and ``unit`` are required for 'user' (the key of the caller's values and
    their unit) and refused otherwise.
    """

    descriptor: str
    coefficient_ppm: float | None
    power: int = 1
    valid_range: tuple[float, float] | None = None
    angle_function: str | None = None
    partner: str | None = None
    name: str | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        put = object.__setattr__
        if self.descriptor not in DESCRIPTORS:
            raise ValueError(f"descriptor {self.descriptor!r} is not one of "
                             f"{DESCRIPTORS}")
        what = f"the {self.descriptor} term"
        if self.coefficient_ppm is not None:
            put(self, "coefficient_ppm",
                _number(self.coefficient_ppm, f"{what}: coefficient_ppm"))
        power = self.power
        if isinstance(power, (bool, np.bool_)) or \
                not isinstance(power, (int, np.integer)) or power == 0:
            raise ValueError(f"{what}: power is {power!r}; a whole number "
                             "other than 0 is needed")
        put(self, "power", int(power))
        if self.valid_range is not None:
            if isinstance(self.valid_range, (str, bytes)) or \
                    len(self.valid_range) != 2:
                raise ValueError(f"{what}: valid_range needs (low, high)")
            low = _number(self.valid_range[0], f"{what}: valid_range low")
            high = _number(self.valid_range[1], f"{what}: valid_range high")
            if low > high:
                raise ValueError(f"{what}: valid_range low {low!r} is above "
                                 f"high {high!r}")
            put(self, "valid_range", (low, high))
        if self.descriptor == "T-O-T angle":
            if self.angle_function not in ANGLE_FUNCTIONS:
                raise ValueError(f"{what}: angle_function needs one of "
                                 f"{ANGLE_FUNCTIONS}, not "
                                 f"{self.angle_function!r}")
        elif self.angle_function is not None:
            raise ValueError(f"{what}: angle_function applies to 'T-O-T "
                             "angle' terms only")
        if self.descriptor in ("bridges to", "bonds to"):
            if self.partner is None:
                raise ValueError(f"{what}: partner (an element) is needed")
            put(self, "partner", validate_symbol(str(self.partner)))
        elif self.descriptor == "mean bond length":
            if self.partner is not None:
                put(self, "partner", validate_symbol(str(self.partner)))
        elif self.partner is not None:
            raise ValueError(f"{what}: partner applies to 'bridges to', "
                             "'bonds to' and 'mean bond length' only")
        if self.descriptor == "user":
            put(self, "name", _text(self.name, f"{what}: name"))
            if not isinstance(self.unit, str):
                raise ValueError(f"{what}: unit needs a text ('1' when "
                                 "dimensionless)")
        elif self.name is not None or self.unit is not None:
            raise ValueError(f"{what}: name and unit apply to 'user' terms "
                             "only")

    def label(self, anion: str = "O") -> str:
        """The descriptor as it appears in tables: 'mean T-O-T angle'."""
        d = self.descriptor
        if d == "T-O-T angle":
            bridge = f"T-{anion}-T"
            return {"deg": f"mean {bridge} angle",
                    "cos": f"mean cos({bridge})",
                    "cos/(cos-1)": f"mean cos/(cos-1) of {bridge}",
                    "sec": f"mean sec({bridge})",
                    "sin(half)": f"mean sin({bridge} / 2)"}[
                        self.angle_function]
        if d == "Qn":
            return f"Qn (bridging {anion})"
        if d == "bridges to":
            return f"bridges to {self.partner} (via {anion})"
        if d == "bonds to":
            return f"bonds to {self.partner}"
        if d == "mean bond length":
            return "mean bond length" + (f" to {self.partner}"
                                         if self.partner else "")
        if d == "user":
            return str(self.name)
        return d

    @property
    def power_label(self) -> str:
        """'' for the first power, '^-3' or '^2' otherwise."""
        return "" if self.power == 1 else f"^{self.power}"

    @property
    def identity(self) -> tuple:
        """What makes two terms the same descriptor, whatever their power."""
        return (self.descriptor, self.partner, self.angle_function,
                self.unit_label)

    @property
    def unit_label(self) -> str:
        """The descriptor's unit: 'deg', 'Å', 'v.u.', '1' or the user's."""
        if self.descriptor == "T-O-T angle":
            return "deg" if self.angle_function == "deg" else "1"
        if self.descriptor == "mean bond length":
            return "Å"
        if self.descriptor == "BVS":
            return "v.u."
        if self.descriptor == "user":
            return str(self.unit)
        return "1"


@dataclass(frozen=True)
class Correlation:
    """A structure-shift correlation, filled in by the user. None ships.

    ``nucleus`` is a mass number and an element symbol ('29Si'); the element
    is the one whose atoms receive a shift (the mass number is a label and is
    not checked against an isotope table). ``intercept_ppm`` and the terms'
    coefficients are the published ones; ``reference`` is the full literature
    reference they come from; ``shift_reference`` the compound the shifts are
    relative to (TMS for 29Si, say), recorded as given. ``anion`` is the
    bridging anion of the 'T-O-T angle', 'Qn' and 'bridges to' descriptors,
    'O' as in the glass analysis.

    Fields may be left None while a correlation is being filled in;
    :meth:`missing` lists what :func:`predict_shifts` still needs, and it
    refuses until nothing is.
    """

    nucleus: str
    intercept_ppm: float | None
    terms: tuple[Term, ...]
    reference: str | None
    shift_reference: str | None = None
    anion: str = "O"
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        put = object.__setattr__
        text = _text(self.nucleus, "nucleus")
        match = _NUCLEUS.match(text)
        if not match or int(match.group(1)) < 1:
            raise ValueError(f"nucleus {self.nucleus!r}: a mass number and an "
                             "element symbol, such as '29Si' or '27Al', are "
                             "needed")
        element = validate_symbol(match.group(2))
        put(self, "nucleus", f"{int(match.group(1))}{element}")
        if self.intercept_ppm is not None:
            put(self, "intercept_ppm",
                _number(self.intercept_ppm, f"{text}: intercept_ppm"))
        if isinstance(self.terms, Term):
            raise ValueError(f"{text}: terms needs a sequence of Term")
        terms = tuple(self.terms)
        for term in terms:
            if not isinstance(term, Term):
                raise ValueError(f"{text}: every term needs to be a Term, not "
                                 f"{type(term).__name__}")
        put(self, "terms", terms)
        for label in ("reference", "shift_reference"):
            value = getattr(self, label)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{text}: {label} needs a text or None")
            if isinstance(value, str):
                put(self, label, value.strip() or None)
        put(self, "anion", validate_symbol(str(self.anion)))
        put(self, "notes", md_stats._notes(self.notes, text))
        seen: dict[str, tuple] = {}
        identity: dict[str, tuple] = {}
        powers = set()
        for term in terms:
            label = term.label(self.anion)
            # one label, one descriptor: the values are kept by label, so a
            # 'user' term named like a computed descriptor would share them
            if label in identity and identity[label] != term.identity:
                raise ValueError(
                    f"{text}: two terms share the label {label!r} but differ "
                    "in descriptor, partner, angle function or unit; a 'user' "
                    "term needs a name that no computed descriptor uses")
            identity[label] = term.identity
            if (label, term.power) in powers:
                raise ValueError(f"{text}: two terms in {label} to the power "
                                 f"{term.power}")
            powers.add((label, term.power))
            if label in seen and seen[label] != term.valid_range:
                raise ValueError(f"{text}: the terms in {label} state "
                                 "different ranges of validity")
            seen[label] = term.valid_range

    @property
    def element(self) -> str:
        return _NUCLEUS.match(self.nucleus).group(2)

    def missing(self) -> tuple[str, ...]:
        """What :func:`predict_shifts` needs and the correlation lacks."""
        out = []
        if self.intercept_ppm is None:
            out.append("the intercept (intercept_ppm)")
        if not self.terms:
            out.append("at least one term")
        for term in self.terms:
            if term.coefficient_ppm is None:
                out.append(f"the coefficient of {term.label(self.anion)}"
                           + term.power_label)
        if not self.reference:
            out.append("the literature reference of the coefficients")
        elif TODO_REFERENCE.lower() in self.reference.lower():
            out.append("the literature reference of the coefficients (the "
                       f"reference given is the placeholder {TODO_REFERENCE!r})")
        return tuple(out)

    def describe(self) -> str:
        """The formula with the user's numbers, '?' where one is missing.

        For a T-O-T angle term: 'delta(29Si) / ppm = a + (b) * [mean T-O-T
        angle / deg]', with the entered intercept and coefficient in place of
        a and b.
        """
        def number(value):
            return "?" if value is None else f"{value!r}"

        parts = [number(self.intercept_ppm)]
        for term in self.terms:
            unit = term.unit_label
            x = f"[{term.label(self.anion)}" + (f" / {unit}]" if unit != "1"
                                                 else "]")
            x += term.power_label
            parts.append(f"({number(term.coefficient_ppm)}) * {x}")
        return f"delta({self.nucleus}) / ppm = " + " + ".join(parts)


def _refusal(correlation: Correlation, missing) -> str:
    return (f"{TODO_REFERENCE}: the {correlation.nucleus} correlation lacks "
            f"{'; '.join(missing)}. FACET ships no correlation coefficients: "
            "enter the published ones together with the reference they come "
            "from")


def _angle_values(function: str, theta_deg: np.ndarray) -> np.ndarray:
    if function == "deg":
        return theta_deg
    radians = np.radians(theta_deg)
    cosine = np.cos(radians)
    with np.errstate(divide="ignore", invalid="ignore"):
        if function == "cos":
            return cosine
        if function == "cos/(cos-1)":
            return np.where(cosine != 1.0, cosine / (cosine - 1.0), np.nan)
        if function == "sec":
            return np.where(cosine != 0.0, 1.0 / cosine, np.nan)
    return np.sin(radians / 2.0)


def _angles_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """atan2(|u x v|, u . v) in degrees, row by row."""
    cross = np.linalg.norm(np.cross(u, v), axis=1)
    dot = np.einsum("ij,ij->i", u, v)
    return np.degrees(np.arctan2(cross, dot))


class _Geometry:
    """Per-atom quantities from one frame's bonds, each computed on first use."""

    def __init__(self, frame: Frame, bonds: Bonds | None,
                 formers: frozenset[str] | None, anion: str):
        self.symbols = frame.elements
        self.n = frame.n_atoms
        self.bonds = bonds
        self.formers = formers
        self.anion = anion
        self._bridges = None
        self._ends = None

    def ends(self):
        """(atom, other, d, v) over both ends of every bond."""
        if self._ends is None:
            b = self.bonds
            self._ends = (np.concatenate([b.cation, b.anion]).astype(np.int64),
                          np.concatenate([b.anion, b.cation]).astype(np.int64),
                          np.concatenate([b.d_ang, b.d_ang]),
                          np.concatenate([b.v_vu, b.v_vu]))
        return self._ends

    def cn(self) -> np.ndarray:
        atom = self.ends()[0]
        return np.bincount(atom, minlength=self.n).astype(np.float64)

    def bvs(self) -> np.ndarray:
        atom, _, _, v = self.ends()
        total = np.bincount(atom, weights=v, minlength=self.n)
        return np.where(self.cn() > 0, total, np.nan)

    def mean_length(self, partner: str | None) -> np.ndarray:
        atom, other, d, _ = self.ends()
        sel = np.ones(atom.shape, dtype=bool) if partner is None else \
            self.symbols[other] == partner
        count = np.bincount(atom[sel], minlength=self.n)
        total = np.bincount(atom[sel], weights=d[sel], minlength=self.n)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(count > 0, total / count, np.nan)

    def bonds_to(self, partner: str) -> np.ndarray:
        atom, other, _, _ = self.ends()
        sel = self.symbols[other] == partner
        return np.bincount(atom[sel], minlength=self.n).astype(np.float64)

    def bridges(self):
        """Ordered pairs (b1, b2) of distinct former bonds to one anion atom.

        Returns (b1, b2, forward, n_formers_on_anion) with forward True for
        one of the two orders of each pair; n_formers is per anion atom.
        """
        if self._bridges is None:
            b = self.bonds
            formers = np.array(sorted(self.formers), dtype="<U2")
            sel = np.flatnonzero(np.isin(self.symbols[b.cation], formers)
                                 & (self.symbols[b.anion] == self.anion))
            anion = b.anion[sel]
            order = np.argsort(anion, kind="stable")
            sel, anion = sel[order], anion[order]
            n_formers = np.bincount(anion, minlength=self.n)
            _, start, count = np.unique(anion, return_index=True,
                                        return_counts=True)
            first, second, forward = [], [], []
            for m in np.unique(count):
                if m < 2:
                    continue
                groups = start[count == m][:, None] + np.arange(m)[None, :]
                pairs = np.array([(p, q) for p in range(m) for q in range(m)
                                  if p != q])
                first.append(sel[groups[:, pairs[:, 0]]].ravel())
                second.append(sel[groups[:, pairs[:, 1]]].ravel())
                forward.append(np.broadcast_to(pairs[:, 0] < pairs[:, 1],
                                               (groups.shape[0],
                                                len(pairs))).ravel())
            empty = np.zeros(0, dtype=np.int64)
            self._bridges = (
                np.concatenate(first) if first else empty,
                np.concatenate(second) if second else empty,
                np.concatenate(forward) if forward else np.zeros(0, bool),
                n_formers)
        return self._bridges

    def qn(self) -> np.ndarray:
        b = self.bonds
        n_formers = self.bridges()[3]
        formers = np.array(sorted(self.formers), dtype="<U2")
        sel = (np.isin(self.symbols[b.cation], formers)
               & (self.symbols[b.anion] == self.anion)
               & (n_formers[b.anion] >= 2))
        return np.bincount(b.cation[sel], minlength=self.n).astype(np.float64)

    def bridges_to(self, partner: str) -> np.ndarray:
        first, second, _, _ = self.bridges()
        cation = self.bonds.cation
        sel = self.symbols[cation[second]] == partner
        return np.bincount(cation[first[sel]],
                           minlength=self.n).astype(np.float64)

    def angle_means(self, function: str, centre: str) -> np.ndarray:
        """Per atom: mean of function(angle) over its bridge angles.

        centre 'former': every ordered pair gives the former of b1 one angle;
        centre 'anion': every unordered pair gives the anion one angle.
        """
        first, second, forward, _ = self.bridges()
        if centre == "anion":
            first, second = first[forward], second[forward]
            owner = self.bonds.anion[first]
        else:
            owner = self.bonds.cation[first]
        vec = self.bonds.vec_ang
        theta = _angles_deg(vec[first], vec[second])
        values = _angle_values(function, theta)
        count = np.bincount(owner, minlength=self.n)
        total = np.bincount(owner, weights=values, minlength=self.n)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(count > 0, total / count, np.nan)


_UNDEFINED_BECAUSE = {
    "T-O-T angle": "no bridge angle",
    "mean bond length": "no bond to the element named",
    "BVS": "no bond",
    "user": "NaN among the values given",
}


def _context_notes(correlation: Correlation, v_bond: float | None,
                   formers: frozenset[str] | None) -> list[str]:
    """What the shifts rest on: the correlation, the bonds, the formers."""
    notes = [
        f"{correlation.nucleus} shifts from the correlation entered by the "
        f"user, {correlation.describe()}; reference: {correlation.reference}; "
        "shifts relative to "
        + (correlation.shift_reference or "a compound not given")]
    if v_bond is not None and any(t.descriptor != "user"
                                  for t in correlation.terms):
        notes.append(f"descriptors from the bonds of bulk.bonds_at at "
                     f"v_bond = {v_bond:g} v.u., the bonds the CN counts")
    if formers is not None:
        notes.append("network formers: "
                     + (", ".join(sorted(formers)) or "none (empty set)")
                     + f"; bridging anion: {correlation.anion}")
    return notes


def _count_notes(correlation: Correlation, n_atoms: int, n_defined: int,
                 undefined_by_term: Mapping[str, int],
                 outside_by_term: Mapping[str, int], over: str) -> list[str]:
    """Atoms without a value, outside a range, or with no range, per term.

    ``over`` is '' for one frame, or ' over N frames (an atom counted once
    per frame)'.
    """
    element = correlation.element
    negative = {t.label(correlation.anion) for t in correlation.terms
                if t.power < 0}
    notes = []
    seen = set()
    for term in correlation.terms:
        label = term.label(correlation.anion)
        if label in seen:
            continue
        seen.add(label)
        if undefined_by_term.get(label, 0):
            reasons = [_UNDEFINED_BECAUSE[term.descriptor]] \
                if term.descriptor in _UNDEFINED_BECAUSE else []
            if label in negative:
                reasons.append("a value of 0, which a negative power leaves "
                               "undefined")
            notes.append(
                f"{undefined_by_term[label]} of {n_atoms} {element} atoms"
                f"{over} have no value of {label} "
                f"({' or '.join(reasons) or 'undefined'}); "
                "their shift is undefined (NaN) and they are in no spectrum")
        if term.valid_range is None:
            notes.append(f"{TODO_REFERENCE}: no range of validity is given "
                         f"for {label}, so no atom is counted outside it")
        elif outside_by_term.get(label, 0):
            low, high = term.valid_range
            notes.append(
                f"{outside_by_term[label]} of {n_defined} {element} atoms "
                f"with a shift{over} have {label} outside the stated range "
                f"[{low!r}, {high!r}] {term.unit_label}; their shifts are the "
                "correlation extrapolated, kept and counted in 'outside'")
    return notes


@dataclass(frozen=True, eq=False)
class ShiftPrediction:
    """Predicted isotropic shifts of every atom of one element in one frame.

    ``rows`` are the atoms' frame rows (ascending), ``atom_id`` their ids.
    ``values`` maps each term's descriptor label to the per-atom value;
    ``shift_ppm`` is NaN where a descriptor is undefined. ``outside`` is True
    for an atom with a shift whose value of some term lies outside that
    term's stated range. The counts per term are in ``undefined_by_term`` and
    ``outside_by_term``.
    """

    correlation: Correlation
    rows: np.ndarray
    atom_id: np.ndarray
    values: Mapping[str, np.ndarray]
    shift_ppm: np.ndarray
    outside: np.ndarray
    undefined_by_term: Mapping[str, int]
    outside_by_term: Mapping[str, int]
    timestep: int | None
    v_bond_vu: float | None
    formers: frozenset[str] | None
    notes: tuple[str, ...]

    @property
    def n_atoms(self) -> int:
        return int(self.rows.shape[0])

    @property
    def n_defined(self) -> int:
        return int(np.isfinite(self.shift_ppm).sum())

    @property
    def n_outside(self) -> int:
        return int(self.outside.sum())

    def as_rows(self) -> list[dict]:
        """One row per atom: id, row, each descriptor value, the shift."""
        units = {t.label(self.correlation.anion): t.unit_label
                 for t in self.correlation.terms}
        rows = []
        for a in range(self.n_atoms):
            row = {"atom id": int(self.atom_id[a]), "row": int(self.rows[a])}
            for label, value in self.values.items():
                unit = units[label]
                row[label + (f" ({unit})" if unit != "1" else "")] = \
                    float(value[a])
            row[f"delta {self.correlation.nucleus} (ppm)"] = \
                float(self.shift_ppm[a])
            row["outside stated range"] = bool(self.outside[a])
            rows.append(row)
        return rows


def _user_array(user_values, name: str, n: int) -> np.ndarray:
    if user_values is None or name not in user_values:
        raise ValueError(f"the 'user' term {name!r} needs user_values[{name!r}]"
                         f", one value per atom of the frame")
    try:
        array = np.array(user_values[name], dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"user_values[{name!r}] are not numbers") from None
    if array.shape != (n,):
        raise ValueError(f"user_values[{name!r}] has shape {array.shape}; "
                         f"({n},), one value per atom of the frame, is needed")
    if np.isinf(array).any():
        raise ValueError(f"user_values[{name!r}] holds an infinite value")
    return array


_BRIDGE_DESCRIPTORS = ("T-O-T angle", "Qn", "bridges to")


def _check_applicable(correlation: Correlation, elements,
                      former_set: frozenset[str] | None) -> None:
    """Refusals that need only the frame's elements and the former set.

    :func:`md_nmr` runs this on its first frame before the first pair
    search; :func:`predict_shifts` on every frame.
    """
    element = correlation.element
    anion = correlation.anion
    present = {str(s) for s in np.unique(elements)}
    if element not in present:
        raise ValueError(f"the frame holds no {element}, the element of "
                         f"{correlation.nucleus}")
    formers_text = ", ".join(sorted(former_set or ())) or "none"
    for term in correlation.terms:
        label = term.label(anion)
        d = term.descriptor
        if d not in _BRIDGE_DESCRIPTORS:
            continue
        if former_set is None:
            raise ValueError(f"{label} depends on the network formers; "
                             "formers is None (no former set is assumed)")
        if anion not in present:
            raise ValueError(f"{label}: the frame holds no {anion}, the "
                             "bridging anion of the correlation, so no bridge "
                             "exists in it")
        if d in ("Qn", "bridges to") and element not in former_set:
            raise ValueError(f"{label} is defined for network formers; "
                             f"{element} is not among the formers given "
                             f"({formers_text})")
        if d == "bridges to" and term.partner not in former_set:
            raise ValueError(
                f"{label}: {term.partner} is not among the formers given "
                f"({formers_text}); a bridge joins two formers through "
                f"{anion}, so bridges to a non-former are not counted and "
                "every value would be 0")
        if d == "T-O-T angle" and element != anion and \
                element not in former_set:
            raise ValueError(
                f"{label} is defined for a network former (the angles of its "
                f"bridges) or for the anion {anion} (the angles centred on "
                f"it); {element} is neither")


def _check_bonds(frame: Frame, bonds) -> None:
    """Refuse bonds that are not this frame's (as md_order checks its own)."""
    if not isinstance(bonds, Bonds):
        raise ValueError(f"bulk.Bonds is needed (bulk.bonds_at), not "
                         f"{type(bonds).__name__}")
    if not len(bonds):
        return
    n = frame.n_atoms
    cation = np.asarray(bonds.cation, np.int64)
    anion = np.asarray(bonds.anion, np.int64)
    if max(int(cation.max()), int(anion.max())) >= n or \
            min(int(cation.min()), int(anion.min())) < 0:
        raise ValueError(f"the bonds name atom rows outside 0..{n - 1}; they "
                         "are another frame's")
    vec, _ = _pair_geometry(frame.frac, frame.box_ang, cation, anion,
                            np.asarray(bonds.image, np.int64))
    worst = float(np.abs(vec - bonds.vec_ang).max())
    if not worst <= _VECTOR_TOL_ANG:
        raise ValueError(
            f"the bond vectors differ by up to {worst:.3g} Å from this "
            "frame's geometry; the bonds were measured on another frame")


def _partner_notes(correlation: Correlation, elements, bonds) -> list[str]:
    """Refuse a bond partner on the bonded element's own side; note an
    absent or unbonded partner, whose values are then 0 (or none)."""
    element = correlation.element
    present = {str(s) for s in np.unique(elements)}
    cations: set[str] = set()
    anions: set[str] = set()
    if bonds is not None and len(bonds):
        cations = {str(s) for s in np.unique(elements[bonds.cation])}
        anions = {str(s) for s in np.unique(elements[bonds.anion])}
    notes = []
    for term in correlation.terms:
        partner = term.partner
        if partner is None:
            continue
        label = term.label(correlation.anion)
        zero = "has no value of" if term.descriptor == "mean bond length" \
            else "has 0 for"
        if partner not in present:
            notes.append(f"the frame holds no {partner}: every {element} "
                         f"atom {zero} {label}")
            continue
        if term.descriptor == "bridges to" or bonds is None:
            continue
        bonded = cations | anions
        if element in bonded and partner in bonded and not (
                (element in cations and partner in anions)
                or (element in anions and partner in cations)):
            side = "cations" if element in cations else "anions"
            raise ValueError(
                f"{label} on {element}: in this frame's bonds {element} and "
                f"{partner} are both bonded only as {side}, and "
                f"bulk.bonds_at bonds a cation to an anion, so no "
                f"{element}-{partner} bond exists")
        if partner not in bonded:
            notes.append(f"no {partner} atom has a bond above v_bond = "
                         f"{bonds.v_bond_vu:g} v.u. in this frame: every "
                         f"{element} atom {zero} {label}")
    return notes


def predict_shifts(correlation: Correlation, frame: Frame,
                   bonds: Bonds | None = None, *, formers,
                   user_values: Mapping[str, Sequence[float]] | None = None
                   ) -> ShiftPrediction:
    """Every atom of the correlation's element, its descriptors and its shift.

    ``bonds`` are the frame's bonds at the CN threshold
    (``bulk.bonds_at``); they may be None only when every term is 'user'.
    ``formers`` has no default: the network formers as a collection of
    element symbols, or None when no term needs them ('T-O-T angle' at a
    former, 'Qn', 'bridges to' refuse None). ``user_values`` maps a 'user'
    term's name to one value per atom of the frame (rows), NaN where
    undefined.

    The bonds are checked to be this frame's (every bond vector formed again
    from the frame's fractions, a difference above 1e-9 Å refused), and a
    descriptor that cannot exist in the frame is refused (module docstring).

    Refused with 'TODO: need reference' when the correlation lacks a
    coefficient, its intercept, any term, or its literature reference.
    """
    if not isinstance(correlation, Correlation):
        raise ValueError(f"a Correlation is needed, not "
                         f"{type(correlation).__name__}")
    missing = correlation.missing()
    if missing:
        raise ValueError(_refusal(correlation, missing))
    frame = _check_frame(frame)
    element = correlation.element
    anion = correlation.anion
    former_set = _former_set(formers, anion)
    _check_applicable(correlation, frame.elements, former_set)
    rows = np.flatnonzero(frame.elements == element)
    if any(t.descriptor != "user" for t in correlation.terms):
        if not isinstance(bonds, Bonds):
            raise ValueError("these terms are computed from the bonds: pass "
                             "bulk.bonds_at(...) of this frame")
    if bonds is not None:
        _check_bonds(frame, bonds)
    geometry = _Geometry(frame, bonds, former_set, anion)
    notes_partner = _partner_notes(correlation, frame.elements, bonds)

    values: dict[str, np.ndarray] = {}
    for term in correlation.terms:
        label = term.label(anion)
        if label in values:
            continue
        d = term.descriptor
        if d == "CN":
            full = geometry.cn()
        elif d == "BVS":
            full = geometry.bvs()
        elif d == "mean bond length":
            full = geometry.mean_length(term.partner)
        elif d == "bonds to":
            full = geometry.bonds_to(term.partner)
        elif d == "Qn":
            full = geometry.qn()
        elif d == "bridges to":
            full = geometry.bridges_to(term.partner)
        elif d == "T-O-T angle":
            full = geometry.angle_means(
                term.angle_function, "anion" if element == anion else "former")
        else:
            full = _user_array(user_values, term.name, frame.n_atoms)
        values[label] = np.array(full[rows], dtype=np.float64)

    undefined = np.zeros(rows.size, dtype=bool)
    no_value = {label: ~np.isfinite(x) for label, x in values.items()}
    for term in correlation.terms:
        if term.power < 0:
            label = term.label(anion)
            no_value[label] = no_value[label] | (values[label] == 0.0)
    undefined_by_term: dict[str, int] = {}
    for label, bad in no_value.items():
        undefined_by_term[label] = int(bad.sum())
        undefined |= bad
    shift = np.full(rows.size, correlation.intercept_ppm)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        for term in correlation.terms:
            shift = shift + term.coefficient_ppm * \
                values[term.label(anion)] ** term.power
    shift[undefined] = np.nan
    if np.isinf(shift).any():
        raise ValueError(f"a predicted {correlation.nucleus} shift overflows "
                         "to infinity")

    outside = np.zeros(rows.size, dtype=bool)
    outside_by_term: dict[str, int] = {}
    term_by_label = {t.label(anion): t for t in correlation.terms}
    for label, x in values.items():
        term = term_by_label[label]
        if term.valid_range is None:
            outside_by_term[label] = 0
            continue
        low, high = term.valid_range
        with np.errstate(invalid="ignore"):
            out = ~undefined & ((x < low) | (x > high))
        outside_by_term[label] = int(out.sum())
        outside |= out
    v_bond = None if bonds is None else float(bonds.v_bond_vu)
    notes = _context_notes(correlation, v_bond, former_set)
    notes += notes_partner
    notes += _count_notes(correlation, rows.size, int((~undefined).sum()),
                          undefined_by_term, outside_by_term, "")
    notes.append(QUADRUPOLAR_NOTE)
    for value in values.values():
        value.setflags(write=False)
    return ShiftPrediction(
        correlation=correlation, rows=md_stats._frozen(rows),
        atom_id=md_stats._frozen(frame.atom_id[rows]), values=values,
        shift_ppm=md_stats._frozen(shift), outside=md_stats._frozen(outside),
        undefined_by_term=undefined_by_term, outside_by_term=outside_by_term,
        timestep=frame.timestep, v_bond_vu=v_bond, formers=former_set,
        notes=tuple(notes))


# ---------------------------------------------------------------------------
# lines on a grid
# ---------------------------------------------------------------------------

def _grid(delta_ppm) -> np.ndarray:
    try:
        grid = np.array(delta_ppm, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("delta_ppm needs numbers") from None
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("delta_ppm needs a 1-D grid of at least two points")
    if not np.isfinite(grid).all():
        raise ValueError("a delta_ppm grid point is NaN or infinite")
    if not (np.diff(grid) > 0).all():
        if (np.diff(grid) < 0).all():
            raise ValueError(
                "delta_ppm has to increase strictly; this grid decreases (the "
                "order NMR plots use): pass it reversed, delta_ppm[::-1], and "
                "a plot can still show ppm decreasing to the right")
        raise ValueError("delta_ppm has to increase strictly")
    return grid


def _lineshape(value) -> str:
    if value not in LINESHAPES:
        raise ValueError(f"lineshape {value!r} is not one of {LINESHAPES}")
    return value


def _shifts(shift_ppm) -> np.ndarray:
    try:
        shifts = np.array(shift_ppm, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("shift_ppm needs numbers") from None
    if not np.isfinite(shifts).all():
        raise ValueError("a shift is NaN or infinite; leave undefined shifts "
                         "out (nmr_spectrum counts them)")
    return shifts


def broadened_spectrum(shift_ppm, delta_ppm, *, fwhm_ppm: float,
                       lineshape: str) -> np.ndarray:
    """One unit-area line per shift, summed on the grid, in atoms per ppm.

    Gaussian: exp(-(delta - s)^2 / (2 sigma^2)) / (sigma sqrt(2 pi)) with
    sigma = FWHM / (2 sqrt(2 ln 2)); Lorentzian:
    (gamma / pi) / ((delta - s)^2 + gamma^2) with gamma = FWHM / 2. Each line
    is evaluated exactly at the grid points (no histogram, no FFT), in
    chunks of :data:`SPECTRUM_CHUNK_ATOMS` sorted shifts; a Gaussian is
    skipped only where it is exactly 0.0 in float64 (beyond 40 sigma).
    """
    grid = _grid(delta_ppm)
    width = _positive(fwhm_ppm, "fwhm_ppm")
    shape = _lineshape(lineshape)
    shifts = np.sort(_shifts(shift_ppm))
    out = np.zeros(grid.size)
    if shape == "gaussian":
        sigma = width / _FWHM_PER_SIGMA
        scale = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
        reach = _GAUSSIAN_ZERO_SIGMAS * sigma
    else:
        gamma = width / 2.0
    for start in range(0, shifts.size, SPECTRUM_CHUNK_ATOMS):
        chunk = shifts[start:start + SPECTRUM_CHUNK_ATOMS]
        if shape == "gaussian":
            # the shifts are sorted, so a chunk spans a narrow range; grid
            # points beyond 40 sigma of every line in it would receive
            # exp(-800), which is exactly 0.0 in float64
            low = int(np.searchsorted(grid, chunk[0] - reach, side="left"))
            high = int(np.searchsorted(grid, chunk[-1] + reach, side="right"))
            if high <= low:
                continue
            buffer = np.subtract.outer(chunk, grid[low:high])
            buffer *= 1.0 / sigma
            np.square(buffer, out=buffer)
            buffer *= -0.5
            np.exp(buffer, out=buffer)
            out[low:high] += scale * buffer.sum(axis=0)
        else:
            buffer = np.subtract.outer(chunk, grid)
            np.square(buffer, out=buffer)
            buffer += gamma * gamma
            np.reciprocal(buffer, out=buffer)
            out += (gamma / math.pi) * buffer.sum(axis=0)
    return out


def line_area_in_grid(shift_ppm, delta_ppm, *, fwhm_ppm: float,
                      lineshape: str) -> float:
    """The exact area of the lines between the first and last grid points.

    The sum over shifts of CDF(last) - CDF(first): erf for the Gaussian,
    arctan for the Lorentzian. It equals the number of shifts when every line
    lies far inside the grid; a Lorentzian always loses some of its tails.
    """
    grid = _grid(delta_ppm)
    width = _positive(fwhm_ppm, "fwhm_ppm")
    shape = _lineshape(lineshape)
    shifts = _shifts(shift_ppm)
    low, high = float(grid[0]), float(grid[-1])
    if shape == "gaussian":
        scale = width / _FWHM_PER_SIGMA * math.sqrt(2.0)
        parts = [0.5 * (math.erf((high - s) / scale)
                        - math.erf((low - s) / scale)) for s in shifts.tolist()]
    else:
        gamma = width / 2.0
        parts = ((np.arctan((high - shifts) / gamma)
                  - np.arctan((low - shifts) / gamma)) / math.pi).tolist()
    return math.fsum(parts)


@dataclass(frozen=True, eq=False)
class NmrSpectrum:
    """A predicted spectrum per frame and over frames, with its bookkeeping.

    ``spectrum``: every atom with a shift, atoms per ppm on ``delta_ppm``.
    ``outside``: the part of it from atoms outside a stated range.
    ``counts``: per frame, atoms 'in range', 'outside range', 'undefined'.
    ``mean_shift``: the mean predicted shift per frame (ppm).
    ``area_numeric`` / ``area_in_grid`` (rows in frame-label order): the
    trapezoid integral of each frame's spectrum, and the exact area of its
    lines inside the grid (:func:`line_area_in_grid`). ``n_below_grid`` /
    ``n_above_grid``: per frame, the shifts below the grid's first point and
    above its last. ``timesteps``: the predictions' frame timesteps (None
    where a frame gave none). ``provenance``: the export header, None when
    the caller gave none (noted).
    """

    correlation: Correlation
    spectrum: Series
    outside: Series
    counts: Distribution
    mean_shift: Scalar
    area_numeric: np.ndarray
    area_in_grid: np.ndarray
    n_below_grid: np.ndarray
    n_above_grid: np.ndarray
    timesteps: tuple
    fwhm_ppm: float
    lineshape: str
    provenance: Provenance | None
    notes: tuple[str, ...]

    def as_sheets(self) -> dict[str, list[dict]]:
        """Tables for ``exporters.write_xlsx``."""
        sheets = {}
        if self.provenance is not None:
            sheets["provenance"] = self.provenance.as_rows()
        sheets["spectrum"] = self.spectrum.as_rows()
        sheets["outside range part"] = self.outside.as_rows()
        sheets["atoms counted"] = self.counts.as_rows(per_frame=True)
        sheets["mean shift"] = self.mean_shift.as_rows(per_frame=True)
        sheets["areas"] = [
            {"frame": int(label), "timestep": step,
             "trapezoid integral (atoms)": float(a),
             "exact area inside the grid (atoms)": float(b),
             "shifts below the grid": int(below),
             "shifts above the grid": int(above)}
            for label, step, a, b, below, above in zip(
                self.spectrum.frames, self.timesteps, self.area_numeric,
                self.area_in_grid, self.n_below_grid, self.n_above_grid)]
        sheets["notes"] = [{"note": note} for note in self.notes]
        return sheets


def _same_correlation(predictions, what: str) -> list:
    """The predictions as a list, refused unless all share one correlation."""
    if isinstance(predictions, ShiftPrediction):
        predictions = [predictions]
    if isinstance(predictions, (str, bytes)):
        raise ValueError(f"{what} takes ShiftPrediction items")
    predictions = list(predictions)
    if not predictions:
        raise ValueError(f"{what}: no prediction given")
    for p in predictions:
        if not isinstance(p, ShiftPrediction):
            raise ValueError(f"{what} takes ShiftPrediction items, not "
                             f"{type(p).__name__}")
    correlation = predictions[0].correlation
    if any(p.correlation != correlation for p in predictions[1:]):
        raise ValueError(f"{what}: the predictions come from different "
                         "correlations (or nuclei); give one correlation's")
    return predictions


def nmr_spectrum(predictions: Sequence[ShiftPrediction], delta_ppm, *,
                 fwhm_ppm: float, lineshape: str,
                 frames: Sequence[int] | None = None,
                 provenance: Provenance | None = None) -> NmrSpectrum:
    """The broadened spectrum of each frame's predictions, averaged.

    Every prediction must come from the same correlation. ``frames`` labels
    the predictions (trajectory indices; 0 .. n-1 when omitted). ``delta_ppm``,
    ``fwhm_ppm`` and ``lineshape`` have no default: they are method choices
    of the user and are stated in the result, with the largest grid step
    against the FWHM and the largest relative difference between the
    trapezoid area and the exact area inside the grid. ``provenance`` (for
    instance ``md_stats.Provenance.from_trajectory`` of the frames the
    predictions came from) is attached for the export; :func:`md_nmr` passes
    its own, and its absence is noted.
    """
    predictions = _same_correlation(predictions, "nmr_spectrum")
    if provenance is not None and not isinstance(provenance, Provenance):
        raise ValueError("provenance needs an md_stats.Provenance or None")
    correlation = predictions[0].correlation
    grid = _grid(delta_ppm)
    width = _positive(fwhm_ppm, "fwhm_ppm")
    shape = _lineshape(lineshape)
    labels = md_stats._frame_labels(frames, len(predictions), "nmr_spectrum")
    order = np.argsort(labels, kind="stable")
    labels = labels[order]
    predictions = [predictions[i] for i in order]

    spectra, parts, counts, means, numeric, exact = [], [], [], [], [], []
    below, above = [], []
    for p in predictions:
        defined = np.isfinite(p.shift_ppm)
        shifts = p.shift_ppm[defined]
        # each line once: the in-range part and the outside part, summed
        part = broadened_spectrum(p.shift_ppm[p.outside], grid,
                                  fwhm_ppm=width, lineshape=shape)
        spectrum = part + broadened_spectrum(
            p.shift_ppm[defined & ~p.outside], grid, fwhm_ppm=width,
            lineshape=shape)
        spectra.append(spectrum)
        parts.append(part)
        counts.append({"in range": int((defined & ~p.outside).sum()),
                       "outside range": int(p.outside.sum()),
                       "undefined": int((~defined).sum())})
        means.append(math.fsum(shifts.tolist()) / shifts.size if shifts.size
                     else float("nan"))
        numeric.append(float(np.trapezoid(spectrum, grid)))
        exact.append(line_area_in_grid(shifts, grid, fwhm_ppm=width,
                                       lineshape=shape))
        below.append(int((shifts < grid[0]).sum()))
        above.append(int((shifts > grid[-1]).sum()))
    nucleus = correlation.nucleus
    step = float(np.diff(grid).max())
    method = (f"{shape} lines of unit area, FWHM {width!r} ppm, on "
              f"{grid.size} points from {float(grid[0])!r} to "
              f"{float(grid[-1])!r} ppm, grid step up to {step!r} ppm "
              f"({step / width:.3g} x the FWHM)")
    first = predictions[0]
    notes = _context_notes(correlation, first.v_bond_vu, first.formers)
    if any(p.v_bond_vu != first.v_bond_vu or p.formers != first.formers
           for p in predictions[1:]):
        notes.append("the frames' predictions differ in v_bond or in the "
                     "former set; each frame's own notes state its values")
    labels_seen = list(first.undefined_by_term)
    undefined = {label: sum(p.undefined_by_term.get(label, 0)
                            for p in predictions) for label in labels_seen}
    outside_counts = {label: sum(p.outside_by_term.get(label, 0)
                                 for p in predictions)
                      for label in labels_seen}
    over = "" if len(predictions) == 1 else \
        f" over {len(predictions)} frames (an atom counted once per frame)"
    notes += _count_notes(correlation, sum(p.n_atoms for p in predictions),
                          sum(p.n_defined for p in predictions), undefined,
                          outside_counts, over)
    notes += [
        method,
        "each atom with a shift contributes one line of unit area, so a "
        "frame's spectrum integrates over all shifts to its number of atoms "
        "with a shift; over the grid it integrates to the area of the lines "
        "inside it, given per frame exactly (area_in_grid) beside the "
        "trapezoid integral (area_numeric)"]
    with_lines = [(a, b) for a, b in zip(numeric, exact) if b > 0.0]
    if with_lines:
        worst = max(abs(a - b) / b for a, b in with_lines)
        notes.append(
            f"the trapezoid integral differs from the exact area inside the "
            f"grid by up to {worst:.3g} (relative) over the frames, a measure "
            "of how finely the grid samples lines of this width")
    if step >= width:
        notes.append(
            f"the largest grid step, {step!r} ppm, is at least the FWHM "
            f"{width!r} ppm: a line can fall between grid points, so the "
            "sampled heights and the trapezoid area depend on where each "
            "line sits between the points")
    if sum(below) or sum(above):
        notes.append(
            f"{sum(below)} predicted shift(s){over} lie below the grid's "
            f"first point {float(grid[0])!r} ppm and {sum(above)} above its "
            f"last point {float(grid[-1])!r} ppm: those atoms are counted as "
            "their descriptors place them, the mean shift includes them, and "
            "only the part of their lines inside the grid is in the spectrum "
            "(area_in_grid holds it exactly)")
    if provenance is None:
        notes.append("no provenance was given, so the trajectory file, the "
                     "frames and the settings behind these predictions are "
                     "not recorded with this spectrum (md_nmr attaches its "
                     "own; pass provenance= when calling nmr_spectrum)")
    notes.append(QUADRUPOLAR_NOTE)
    spectrum = Series.from_frames(
        grid, spectra, name=f"{nucleus} spectrum", axis_name="delta_ppm",
        axis_unit="ppm", value_unit="atoms/ppm", frames=labels.tolist(),
        notes=(method,))
    outside = Series.from_frames(
        grid, parts, name=f"{nucleus} spectrum, atoms outside the stated "
                          "range", axis_name="delta_ppm", axis_unit="ppm",
        value_unit="atoms/ppm", frames=labels.tolist(), notes=(method,))
    count_dist = Distribution.from_counts(
        counts, name=f"{correlation.element} atoms ({nucleus})", kind="count",
        keys=("in range", "outside range", "undefined"),
        frames=labels.tolist())
    mean_shift = Scalar(f"mean predicted delta {nucleus}", "ppm", means,
                        frames=labels.tolist())
    return NmrSpectrum(
        correlation=correlation, spectrum=spectrum, outside=outside,
        counts=count_dist, mean_shift=mean_shift,
        area_numeric=md_stats._frozen(np.array(numeric)),
        area_in_grid=md_stats._frozen(np.array(exact)),
        n_below_grid=md_stats._frozen(np.array(below, dtype=np.int64)),
        n_above_grid=md_stats._frozen(np.array(above, dtype=np.int64)),
        timesteps=tuple(p.timestep for p in predictions), fwhm_ppm=width,
        lineshape=shape, provenance=provenance, notes=tuple(notes))


def shift_histogram(predictions: Sequence[ShiftPrediction], edges_ppm, *,
                    density: bool = False,
                    frames: Sequence[int] | None = None) -> Histogram:
    """The predicted shifts binned on ``edges_ppm``, per frame and averaged.

    Every prediction must come from the same correlation, as for
    :func:`nmr_spectrum`. Undefined shifts are counted in ``n_nonfinite``,
    shifts beyond the edges in ``n_below`` / ``n_above``
    (``md_stats.Histogram``).
    """
    predictions = _same_correlation(predictions, "shift_histogram")
    nucleus = predictions[0].correlation.nucleus
    return Histogram.from_samples(
        [p.shift_ppm for p in predictions], edges_ppm,
        name=f"predicted delta {nucleus}", unit="ppm", density=density,
        frames=frames)


def md_nmr(trajectory, ox: ModelOxidation, correlation: Correlation, *,
           formers, delta_ppm, fwhm_ppm: float, lineshape: str,
           frames: Sequence[int] | None = None,
           params: bv.ParameterSet | None = None,
           v_bond_vu: float = bv.V_BOND_DEFAULT,
           v_list_vu: float = bv.V_LIST_DEFAULT,
           r_search_ang: float | None = None, method: str = "auto",
           progress: Callable[[int, int], None] | None = None,
           cancelled: Callable[[], bool] | None = None) -> NmrSpectrum:
    """Predicted spectrum over a trajectory: one pair search per frame.

    Per frame: ``bulk.analyse_frame`` (the one search), ``bulk.bonds_at`` at
    ``v_bond_vu``, :func:`predict_shifts`; then :func:`nmr_spectrum` over the
    frames, with a provenance header. A correlation with a 'user' term is
    refused here (its values are per frame; call :func:`predict_shifts` with
    them and :func:`nmr_spectrum`). The correlation, the grid and the former
    set are checked before any frame is read, and the first frame's
    elements (the nucleus, the bridging anion, the formers a term needs)
    before its pair search.
    """
    if not isinstance(correlation, Correlation):
        raise ValueError("a Correlation is needed")
    missing = correlation.missing()
    if missing:
        raise ValueError(_refusal(correlation, missing))
    if any(t.descriptor == "user" for t in correlation.terms):
        raise ValueError("md_nmr computes every descriptor from the bonds; a "
                         "'user' term needs predict_shifts with its values "
                         "for each frame")
    if not isinstance(ox, ModelOxidation):
        raise ValueError("ox needs a ModelOxidation (md_model.model_oxidation)")
    grid = _grid(delta_ppm)
    width = _positive(fwhm_ppm, "fwhm_ppm")
    shape = _lineshape(lineshape)
    former_set = _former_set(formers, correlation.anion)
    indices = _frame_indices(trajectory, frames)
    params = params or bv.DEFAULT
    predictions = []
    estimated: dict[str, int] = {}
    missing_pairs: dict[str, int] = {}
    radii = set()
    for done, k in enumerate(indices):
        _tick(progress, cancelled, done, len(indices))
        frame = trajectory.frame(k)
        if done == 0:
            _check_applicable(correlation, frame.elements, former_set)
        table, _ = analyse_frame(frame, ox.per_atom(frame.elements), params,
                                 v_bond_vu=v_bond_vu, v_list_vu=v_list_vu,
                                 r_search_ang=r_search_ang, method=method)
        radii.add(table.r_search_ang)
        for target, source in ((estimated, table.estimated_pairs),
                               (missing_pairs, table.missing_pairs)):
            for pair, count in source.items():
                target[pair] = target.get(pair, 0) + int(count)
        predictions.append(predict_shifts(
            correlation, frame, bonds_at(table, v_bond_vu),
            formers=former_set))
    _tick(progress, cancelled, len(indices), len(indices))
    notes = []
    if len(radii) > 1:
        notes.append(f"the search radius varied between frames: "
                     f"{sorted(radii)} Å")
    if estimated or missing_pairs:
        notes.append("the estimated and missing bond-valence pair counts "
                     "below are summed over the frames used")
    provenance = Provenance.from_trajectory(
        trajectory, frames_used=indices, params=params, ox=ox,
        v_bond_vu=v_bond_vu, v_list_vu=v_list_vu,
        r_search_ang=max(radii), formers=former_set,
        method_parameters={"lineshape": shape, "fwhm_ppm": width,
                           "delta_ppm first": float(grid[0]),
                           "delta_ppm last": float(grid[-1]),
                           "delta_ppm points": int(grid.size),
                           "pair search method": method,
                           "correlation": correlation.describe(),
                           "correlation reference": correlation.reference},
        estimated_pairs=estimated, missing_pairs=missing_pairs,
        notes=tuple(notes))
    result = nmr_spectrum(predictions, grid, fwhm_ppm=width, lineshape=shape,
                          frames=indices, provenance=provenance)
    return dataclasses.replace(result, notes=result.notes + tuple(notes))


# ===========================================================================
# 3. EXAFS
# ===========================================================================

@dataclass(frozen=True, eq=False)
class AbsorberPairs:
    """Every pair of one frame's search centred on an atom of the absorber.

    ``i`` (absorber row), ``j``, ``image``, ``d_ang`` and ``vec_ang`` (None
    when the search kept no vectors) are in the order the search gave them;
    nothing downstream depends on that order (:func:`feff_cluster` sorts its
    own atoms by distance, the sums use ``math.fsum``, the partials sort).
    ``r_ang`` is the radius the pairs were searched to; ``vectors_within_ang``
    the radius within which vectors were kept (None: all). The frame's
    elements, ids, volume and widths travel with them, so the functions below
    need nothing else.
    """

    absorber: str
    rows: np.ndarray
    i: np.ndarray
    j: np.ndarray
    image: np.ndarray
    d_ang: np.ndarray
    vec_ang: np.ndarray | None
    r_ang: float
    d_min_ang: float
    vectors_within_ang: float | None
    n_below_d_min: int
    elements: np.ndarray
    atom_id: np.ndarray
    volume_ang3: float
    half_width_ang: float
    timestep: int | None

    @property
    def n_absorbers(self) -> int:
        return int(self.rows.shape[0])

    def count(self, element: str) -> int:
        """Atoms of ``element`` in the frame."""
        return int((self.elements == element).sum())


def absorber_pairs(frame: Frame, pairs, absorber: str) -> AbsorberPairs:
    """The pairs of the frame's one search that start on an absorber atom.

    ``pairs`` is a ``bulk.PairTable`` or the blocks of ``bulk.iter_pairs``;
    they are checked, as ``bulk.valence_table`` checks them, to come from
    this frame and to cover every centre atom exactly once.
    """
    frame = _check_frame(frame)
    absorber = validate_symbol(str(absorber))
    mask = frame.elements == absorber
    if not mask.any():
        raise ValueError(f"the frame holds no {absorber}")
    coverage = _Coverage(frame, frame.n_atoms)
    parts = []
    n_below = 0
    for block in _blocks_of(pairs):
        coverage.add(block)
        keep = mask[block.i]
        parts.append((block.i[keep], block.j[keep], block.image[keep],
                      block.d_ang[keep],
                      None if block.vec_ang is None else block.vec_ang[keep]))
        n_below += int(block.n_below_d_min)
    coverage.check()
    first: PairTable = coverage.first
    i = np.concatenate([p[0] for p in parts]).astype(np.int64)
    j = np.concatenate([p[1] for p in parts]).astype(np.int64)
    image = np.concatenate([p[2] for p in parts])
    d = np.concatenate([p[3] for p in parts])
    vec = None if parts[0][4] is None else np.concatenate([p[4] for p in parts])
    frozen = md_stats._frozen
    return AbsorberPairs(
        absorber=absorber, rows=frozen(np.flatnonzero(mask)),
        i=frozen(i), j=frozen(j), image=frozen(image), d_ang=frozen(d),
        vec_ang=None if vec is None else frozen(vec),
        r_ang=float(first.r_ang), d_min_ang=float(first.d_min_ang),
        vectors_within_ang=first.vectors_within_ang, n_below_d_min=n_below,
        elements=frame.elements, atom_id=frame.atom_id,
        volume_ang3=frame.volume_ang3,
        half_width_ang=float(frame.perpendicular_widths_ang.min() / 2.0),
        timestep=frame.timestep)


def _within_half_width(r_ang: float, half_width_ang: float, what: str) -> None:
    """Refuse a radius past half the smallest perpendicular box width."""
    if r_ang > half_width_ang:
        raise ValueError(
            f"{what} {r_ang} is beyond half the smallest perpendicular width "
            f"of the box ({half_width_ang!r} Å), where the model's periodic "
            "images start to correlate with themselves (a pair is then "
            f"counted once per image); give {what} up to that")


def _rdf_points(r_max_ang, dr_ang) -> tuple[np.ndarray, float]:
    """pdf.pair_distribution's grid, arange(dr, r_max + dr/2, dr), and the
    radius its pairs have to be searched to: the last point plus dr, since
    a pair up to one step past the last point deposits part of its weight
    on it (``glass.partial_rdf`` asks the same)."""
    r_max = _positive(r_max_ang, "r_max_ang")
    dr = _positive(dr_ang, "dr_ang")
    if dr >= r_max:
        raise ValueError(f"dr_ang {dr} is not below r_max_ang {r_max}")
    grid = np.arange(dr, r_max + 0.5 * dr, dr)
    return grid, float(grid[-1] + dr)


def _rdf_grid(r_max_ang, dr_ang, ap: AbsorberPairs) -> np.ndarray:
    grid, need = _rdf_points(r_max_ang, dr_ang)
    r_max = float(r_max_ang)
    if need > ap.r_ang + _GRID_ROUNDING_ANG:
        raise ValueError(
            f"the pairs were searched to {ap.r_ang} Å; a g(r) grid to "
            f"r_max_ang {r_max} needs them to {need!r} Å, its last point plus "
            "dr_ang, because a pair up to one step past the last point "
            "deposits part of its weight on it")
    _within_half_width(r_max, ap.half_width_ang, "r_max_ang")
    return grid


@dataclass(frozen=True, eq=False)
class AbsorberRDF:
    """g_AB(r), R_AB(r) and N_AB(r) about absorber A, one frame.

    ``r_ang`` is pdf.pair_distribution's grid, arange(dr, r_max + dr/2, dr).
    ``rdf_per_ang`` is R_AB: neighbours per absorber per Å (the deposited
    histogram / (N_A dr)); ``g`` = R_AB / (4 pi r^2 rho_B), rho_B = N_B / V;
    ``n_cum`` the exact count of B within r of an absorber, per absorber.
    """

    absorber: str
    neighbour: str
    r_ang: np.ndarray
    g: np.ndarray
    rdf_per_ang: np.ndarray
    n_cum: np.ndarray
    n_absorbers: int
    n_neighbours: int
    rho_neighbour_per_ang3: float
    dr_ang: float


def absorber_rdf(ap: AbsorberPairs, *, r_max_ang: float, dr_ang: float,
                 neighbours: Sequence[str] | None = None
                 ) -> dict[str, AbsorberRDF]:
    """The absorber-centred partials of one frame, one per neighbour element.

    ``neighbours=None`` takes every element of the frame, the absorber's own
    included. ``r_max_ang`` and ``dr_ang`` have no default (method choices).
    Refused when the pairs were searched to less than the grid's last point
    plus ``dr_ang`` (a pair up to one step past the last point deposits on
    it, so a shorter search leaves the last point short of those pairs), and
    when r_max lies beyond half the smallest perpendicular box width.
    """
    if not isinstance(ap, AbsorberPairs):
        raise ValueError("absorber_rdf needs AbsorberPairs (absorber_pairs)")
    r = _rdf_grid(r_max_ang, dr_ang, ap)
    dr = float(dr_ang)
    present = sorted(set(str(s) for s in np.unique(ap.elements)))
    if neighbours is None:
        chosen = present
    else:
        if isinstance(neighbours, (str, bytes)):
            raise ValueError("neighbours needs a collection of symbols")
        chosen = [validate_symbol(str(s)) for s in neighbours]
        absent = [s for s in chosen if s not in present]
        if absent:
            raise ValueError(f"the frame holds no {', '.join(absent)}")
    n_abs = ap.n_absorbers
    symbols = ap.elements[ap.j]
    out = {}
    for element in chosen:
        # every distance that deposits on the grid: up to one step past it
        sel = (symbols == element) & (ap.d_ang < r[-1] + dr)
        distances = np.sort(ap.d_ang[sel])
        deposit = _deposit(r, distances, np.ones(distances.size))
        rdf = deposit / (n_abs * dr)
        n_b = ap.count(element)
        rho = n_b / ap.volume_ang3
        g = rdf / (4.0 * math.pi * r * r * rho)
        n_cum = np.searchsorted(distances, r, side="right") / n_abs
        frozen = md_stats._frozen
        out[element] = AbsorberRDF(
            absorber=ap.absorber, neighbour=element, r_ang=frozen(r),
            g=frozen(g), rdf_per_ang=frozen(rdf), n_cum=frozen(n_cum),
            n_absorbers=n_abs, n_neighbours=n_b, rho_neighbour_per_ang3=rho,
            dr_ang=dr)
    return out


@dataclass(frozen=True)
class ShellLimit:
    """Where a shell ends, and how that was decided.

    ``r_ang`` None means no limit (``reason`` says why). ``g_at_limit`` is
    g(r) as computed (not smoothed) at the limit; ``r_peak_ang`` and
    ``g_peak`` the highest point of g below the limit, the first shell's
    maximum. ``level`` is the uncorrelated level g was compared with,
    ``floor_r_ang`` the first and last grid point of the floor the limit was
    taken from, ``g_std_error`` the counting error of g at the limit and
    ``depth_std_errors`` how far the floor lies below the level in such
    errors (``glass.RdfMinimum``; None where not measured). ``notes`` state
    what a reader of the limit needs to know, such as g at the limit not
    being below the level.
    """

    absorber: str
    neighbour: str
    r_ang: float | None
    g_at_limit: float | None
    r_peak_ang: float | None
    source: str
    method: str
    g_peak: float | None = None
    reason: str = ""
    level: float | None = None
    floor_r_ang: tuple[float, float] | None = None
    g_std_error: float | None = None
    depth_std_errors: float | None = None
    notes: tuple[str, ...] = ()


def first_minimum(r_ang, g, *, method: glass.MinimumMethod,
                  absorber: str = "", neighbour: str = "",
                  level: float = 1.0, pairs_per_unit_g=None) -> ShellLimit:
    """The first-shell limit of a g(r), by ``glass.first_minimum``.

    ``method`` is a ``glass.MinimumMethod`` (rule 'valley' or 'first local
    minimum', smoothing, flat rule, margin in standard errors; no default),
    so that a g(r) gets the one cutoff the glass analysis would give it: the
    located point is a grid point, the rule is stated in
    ``ShellLimit.method``, and ``level`` (the uncorrelated level, 1 for two
    elements, 1 - 1/N_A for one) and ``pairs_per_unit_g``
    (``glass.pair_counts_per_unit_g``, from which g's counting errors come;
    None takes g as exact) are glass's. :func:`md_exafs` passes both from
    the model's counts. This adds the highest point of g below the limit,
    and a note when g at the limit is not below the level: a rule that
    compares values as they stand can stop at a noise dip inside a peak, and
    the note gives both heights so that the cut can be read off. A second
    note is added when g beyond the limit rises above the highest point
    below it: a rule can also stop before the main peak, after a rise at its
    foot, with g at the limit below the level (Al-Na of a real glass at
    margin 0, module docstring), and the note gives both maxima.
    """
    if not isinstance(method, glass.MinimumMethod):
        raise ValueError("method needs a glass.MinimumMethod (rule, "
                         "smooth_sigma_ang, flat_rule, margin_std_errors; no "
                         "default rule)")
    found = glass.first_minimum(r_ang, g, method, pair=(absorber, neighbour),
                                level=level, pairs_per_unit_g=pairs_per_unit_g)
    if not found.found:
        return ShellLimit(absorber, neighbour, None, None, None, "auto",
                          found.method, reason=found.reason,
                          level=found.level, notes=tuple(found.notes))
    r = np.asarray(r_ang, dtype=np.float64)
    values = np.asarray(g, dtype=np.float64)
    r_peak = g_peak = None
    notes = list(found.notes)
    pair = f"{absorber}-{neighbour}: " if absorber else ""
    if found.index > 0:
        p = int(np.argmax(values[:found.index]))
        r_peak, g_peak = float(r[p]), float(values[p])
        q = found.index + int(np.argmax(values[found.index:]))
        if values[q] > g_peak:
            notes.append(
                f"{pair}g(r) beyond the limit {found.r_ang!r} Å reaches "
                f"{float(values[q]):.4g} at {float(r[q])!r} Å, above "
                f"{g_peak:.4g}, the highest point below the limit (at "
                f"{r_peak!r} Å): the shell inside the limit holds a lower "
                f"maximum than g reaches after it ({method.rule} rule, "
                f"margin {method.margin_std_errors:g} standard errors)")
    if found.g_value >= found.level:
        notes.append(
            f"{pair}g(r) at the limit {found.r_ang!r} Å is "
            f"{found.g_value:.4g}, not below its uncorrelated level "
            f"{found.level:.6g}"
            + ("" if r_peak is None else
               f"; the highest point of g below the limit is {g_peak:.4g} at "
               f"{r_peak!r} Å, {found.index - p} grid step(s) inside it")
            + f"; the limit cuts the shell where g is that high "
            f"({method.rule} rule, margin {method.margin_std_errors:g} "
            "standard errors)")
    return ShellLimit(absorber, neighbour, found.r_ang, found.g_value,
                      r_peak, "auto", found.method, g_peak=g_peak,
                      level=found.level, floor_r_ang=found.floor_r_ang,
                      g_std_error=found.g_std_error,
                      depth_std_errors=found.depth_std_errors,
                      notes=tuple(notes))


def distance_cumulants(d_ang, weights=None) -> tuple[float, float, float,
                                                      float]:
    """(C1, C2, C3, C4) of a distance distribution, population cumulants.

    C1 the mean, C2 the variance (sigma^2), C3 the third central moment,
    C4 = mu4 - 3 mu2^2. Two passes (the mean first, then the central moments
    from the deviations), every sum with ``math.fsum``. ``weights`` (one per
    distance, 0 or more, not all 0) weight each distance. No distance gives
    four NaN.
    """
    try:
        d = np.array(d_ang, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("d_ang needs numbers") from None
    if not np.isfinite(d).all():
        raise ValueError("a distance is NaN or infinite")
    if d.size == 0:
        return (float("nan"),) * 4
    if weights is None:
        w = np.ones(d.size)
    else:
        try:
            w = np.array(weights, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("weights need numbers") from None
        if w.shape != d.shape:
            raise ValueError("one weight per distance is needed")
        if not np.isfinite(w).all() or (w < 0).any():
            raise ValueError("a weight is negative, NaN or infinite")
    total = math.fsum(w.tolist())
    if total <= 0.0:
        raise ValueError("the weights sum to 0")
    c1 = math.fsum((w * d).tolist()) / total
    e = d - c1
    e2 = e * e
    mu2 = math.fsum((w * e2).tolist()) / total
    mu3 = math.fsum((w * e2 * e).tolist()) / total
    mu4 = math.fsum((w * e2 * e2).tolist()) / total
    return c1, mu2, mu3, mu4 - 3.0 * mu2 * mu2


def _weighting(value) -> str:
    if value not in WEIGHTINGS:
        raise ValueError(f"weighting {value!r} is not one of {WEIGHTINGS}")
    return value


@dataclass(frozen=True, eq=False)
class ShellCumulants:
    """One shell, one frame: N, R = C1, sigma^2 = C2, C3, C4.

    The shell holds the absorber-neighbour pairs with
    ``r_lo_ang < d <= r_hi_ang``. N counts them per absorber, unweighted;
    C1-C4 are cumulants of their distances under ``weighting``.
    ``d_ang`` keeps the distances, so frames can be pooled.
    """

    absorber: str
    neighbour: str
    r_lo_ang: float
    r_hi_ang: float
    weighting: str
    n_absorbers: int
    n_pairs: int
    n_per_absorber: float
    r_mean_ang: float
    sigma2_ang2: float
    c3_ang3: float
    c4_ang4: float
    d_ang: np.ndarray
    timestep: int | None


def shell_cumulants(ap: AbsorberPairs, neighbour: str, *, r_lo_ang: float,
                    r_hi_ang: float, weighting: str) -> ShellCumulants:
    """N and the cumulants of the absorber-``neighbour`` distances in a shell.

    ``r_hi_ang`` is refused beyond the search radius, and beyond half the
    smallest perpendicular box width, as r_max of :func:`absorber_rdf` is.
    ``weighting`` is 'none' (the distribution of distances itself) or
    '1/r^2' (each distance weighted by 1/d^2, the geometric factor of the
    EXAFS amplitude).
    """
    if not isinstance(ap, AbsorberPairs):
        raise ValueError("shell_cumulants needs AbsorberPairs (absorber_pairs)")
    neighbour = validate_symbol(str(neighbour))
    low = _number(r_lo_ang, "r_lo_ang")
    high = _positive(r_hi_ang, "r_hi_ang")
    if low < 0.0 or low >= high:
        raise ValueError(f"the shell [{low}, {high}] Å needs 0 <= r_lo < r_hi")
    if high > ap.r_ang:
        raise ValueError(f"r_hi_ang {high} is beyond the {ap.r_ang} Å the "
                         "pairs were searched to")
    _within_half_width(high, ap.half_width_ang, "r_hi_ang")
    weighting = _weighting(weighting)
    sel = (ap.elements[ap.j] == neighbour) & (ap.d_ang > low) & \
        (ap.d_ang <= high)
    d = np.array(ap.d_ang[sel])
    weights = None if weighting == "none" else 1.0 / (d * d)
    c1, c2, c3, c4 = distance_cumulants(d, weights)
    d.setflags(write=False)
    return ShellCumulants(
        absorber=ap.absorber, neighbour=neighbour, r_lo_ang=low,
        r_hi_ang=high, weighting=weighting, n_absorbers=ap.n_absorbers,
        n_pairs=int(d.size), n_per_absorber=d.size / ap.n_absorbers,
        r_mean_ang=c1, sigma2_ang2=c2, c3_ang3=c3, c4_ang4=c4, d_ang=d,
        timestep=ap.timestep)


_CUMULANT_FIELDS = (("N", "1", "n_per_absorber"),
                    ("R", "Å", "r_mean_ang"),
                    ("sigma^2", "Å^2", "sigma2_ang2"),
                    ("C3", "Å^3", "c3_ang3"),
                    ("C4", "Å^4", "c4_ang4"))


@dataclass(frozen=True, eq=False)
class AveragedShell:
    """One shell over frames: per-frame mean and spread, and the pooled values.

    ``per_frame`` maps 'N', 'R', 'sigma^2', 'C3', 'C4' to ``md_stats.Scalar``
    (mean and spread across frames). ``pooled`` holds the same five over
    every pair of every frame together, which is the average an EXAFS
    measurement makes: its sigma^2 includes the frame-to-frame variation of
    the mean distance, the per-frame mean does not.
    """

    absorber: str
    neighbour: str
    r_lo_ang: float
    r_hi_ang: float
    weighting: str
    limit_source: str
    per_frame: Mapping[str, Scalar]
    pooled: Mapping[str, float]
    n_pairs_total: int
    notes: tuple[str, ...]

    def as_rows(self) -> list[dict]:
        rows = []
        for label, unit, _ in _CUMULANT_FIELDS:
            scalar = self.per_frame[label]
            rows.append({
                "absorber": self.absorber, "neighbour": self.neighbour,
                "shell low (Å)": self.r_lo_ang,
                "shell high (Å)": self.r_hi_ang,
                "limit source": self.limit_source,
                "weighting": self.weighting, "quantity": label, "unit": unit,
                "mean over frames": scalar.mean,
                "std across frames": scalar.std,
                "pooled over frames": self.pooled[label],
                "frames": scalar.n_frames})
        return rows


def average_shells(per_frame: Sequence[ShellCumulants], *,
                   frames: Sequence[int] | None = None,
                   limit_source: str = "user") -> AveragedShell:
    """Per-frame cumulants of one shell, averaged and pooled.

    Every item has the same absorber, neighbour, limits and weighting.
    ``limit_source`` says where the limits came from, for the export.
    """
    items = list(per_frame)
    if not items:
        raise ValueError("no frame to average")
    for item in items:
        if not isinstance(item, ShellCumulants):
            raise ValueError("average_shells takes ShellCumulants items")
    first = items[0]
    signature = (first.absorber, first.neighbour, first.r_lo_ang,
                 first.r_hi_ang, first.weighting)
    if any((x.absorber, x.neighbour, x.r_lo_ang, x.r_hi_ang, x.weighting)
           != signature for x in items[1:]):
        raise ValueError("the frames' shells differ in absorber, neighbour, "
                         "limits or weighting")
    name = f"{first.absorber}-{first.neighbour} shell " \
           f"({first.r_lo_ang:g}, {first.r_hi_ang:g}] Å"
    scalars = {}
    for label, unit, attribute in _CUMULANT_FIELDS:
        scalars[label] = Scalar(f"{name}: {label}", unit,
                                [getattr(x, attribute) for x in items],
                                frames=frames)
    d = np.concatenate([x.d_ang for x in items])
    weights = None if first.weighting == "none" else 1.0 / (d * d)
    c1, c2, c3, c4 = distance_cumulants(d, weights)
    n_abs = sum(x.n_absorbers for x in items)
    pooled = {"N": d.size / n_abs, "R": c1, "sigma^2": c2, "C3": c3,
              "C4": c4}
    notes = [
        f"{name}: N counts the {first.neighbour} atoms with "
        f"{first.r_lo_ang:g} < d <= {first.r_hi_ang:g} Å per "
        f"{first.absorber} atom; R, sigma^2, C3, C4 are population cumulants "
        "of their distances"
        + (" weighted by 1/d^2" if first.weighting == "1/r^2" else
           " (unweighted: the distribution itself, not the effective "
           "distribution an EXAFS fit returns)"),
        "pooled: every pair of every frame as one distribution; mean over "
        "frames: the per-frame values averaged, with their spread across "
        "frames (ddof = 1, not a standard error)"]
    if any(x.n_pairs == 0 for x in items):
        notes.append(f"{name}: {sum(x.n_pairs == 0 for x in items)} frame(s) "
                     "hold no pair in the shell; their cumulants are NaN")
    return AveragedShell(
        absorber=first.absorber, neighbour=first.neighbour,
        r_lo_ang=first.r_lo_ang, r_hi_ang=first.r_hi_ang,
        weighting=first.weighting, limit_source=limit_source,
        per_frame=scalars, pooled=pooled, n_pairs_total=int(d.size),
        notes=tuple(notes))


# ---------------------------------------------------------------------------
# FEFF: clusters out, chi(k) back
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class FeffCluster:
    """One absorber and its neighbours within a radius, from one frame.

    ``vec_ang`` (n, 3) points from the absorber to each neighbour image,
    sorted by distance (``d_ang``); ``elements`` are theirs.
    """

    absorber: str
    absorber_row: int
    absorber_id: int
    frame_label: int | None
    timestep: int | None
    elements: tuple[str, ...]
    vec_ang: np.ndarray
    d_ang: np.ndarray
    cluster_radius_ang: float

    @property
    def label(self) -> str:
        return f"{self.absorber}{self.absorber_id}"

    @property
    def n_potentials(self) -> int:
        """The POTENTIALS entries :func:`write_feff_input` gives it: ipot 0
        for the absorber, then one per element in the cluster."""
        return 1 + len(set(self.elements))


def feff_cluster(ap: AbsorberPairs, row: int, *, cluster_radius_ang: float,
                 frame_label: int | None = None) -> FeffCluster:
    """The cluster about absorber atom ``row``: every pair with d <= radius.

    The pairs need their vectors out to the radius (``iter_pairs`` with
    ``vectors_within_ang`` None or at least the radius); a radius beyond the
    search radius is refused. Inclusive at the radius, as
    ``NeighborFinder`` (and so ``exporters.write_feff``) is.
    """
    if not isinstance(ap, AbsorberPairs):
        raise ValueError("feff_cluster needs AbsorberPairs (absorber_pairs)")
    row = _whole(row, "row", 0)
    if row not in set(ap.rows.tolist()):
        raise ValueError(f"row {row} is not a {ap.absorber} atom")
    radius = _positive(cluster_radius_ang, "cluster_radius_ang")
    if radius > ap.r_ang:
        raise ValueError(f"cluster_radius_ang {radius} is beyond the "
                         f"{ap.r_ang} Å the pairs were searched to")
    if ap.vec_ang is None or (ap.vectors_within_ang is not None
                              and ap.vectors_within_ang < radius):
        raise ValueError("the pairs carry no vectors out to the cluster "
                         "radius; search with vectors_within_ang >= "
                         "cluster_radius_ang")
    sel = np.flatnonzero((ap.i == row) & (ap.d_ang <= radius))
    order = sel[np.argsort(ap.d_ang[sel], kind="stable")]
    return FeffCluster(
        absorber=ap.absorber, absorber_row=row,
        absorber_id=int(ap.atom_id[row]), frame_label=frame_label,
        timestep=ap.timestep,
        elements=tuple(str(s) for s in ap.elements[ap.j[order]]),
        vec_ang=md_stats._frozen(ap.vec_ang[order]),
        d_ang=md_stats._frozen(ap.d_ang[order]), cluster_radius_ang=radius)


def _edge(value) -> str:
    text = _text(value, "edge")
    if not _EDGE.match(text):
        raise ValueError(f"edge {value!r}: one of K, L1-L3, M1-M5, N1-N7")
    return text


def _r_path(r_path_ang, radius: float) -> float:
    # exporters.write_feff's rule when none is given: 1 Å inside the cluster,
    # at least 2 Å
    r_path = max(radius - 1.0, 2.0) if r_path_ang is None else \
        _positive(r_path_ang, "r_path_ang")
    if r_path >= radius:
        raise ValueError(f"RPATH {r_path} Å is not inside the cluster radius "
                         f"{radius} Å; the atoms at the cluster's edge carry "
                         "the least complete potentials, so RPATH stays "
                         "inside it")
    return r_path


def write_feff_input(cluster: FeffCluster, path: str | Path, *, edge: str,
                     r_path_ang: float | None = None,
                     source: str | None = None,
                     extra_comments: Sequence[str] = ()) -> Path:
    """``feff.inp`` for one cluster, in ``exporters.write_feff``'s format.

    The same cards and the same three rules found by running FEFF8L on
    FACET's crystal inputs: the absorber alone has ipot 0 and the other atoms
    of its element get a potential of their own; ``PRINT 0 0 0 0 0 3`` so
    that FEFF writes ``files.dat`` and the ``feffNNNN.dat`` path files; RPATH
    inside the cluster (default: the cluster radius - 1 Å, at least 2 Å,
    write_feff's rule). Potentials are numbered by first appearance in
    distance order, as write_feff numbers them. ``edge`` has no default.

    The comment header names the FACET version and the time of writing, the
    trajectory file (``source``, when given), the frame, its timestep, and
    the absorber's id and row. It states no bond-valence setting: the
    cluster is cut by distance alone. A cluster that needs more unique
    potentials than the FEFF8L FACET was checked with accepts (10) carries a
    comment saying so.
    """
    if not isinstance(cluster, FeffCluster):
        raise ValueError("write_feff_input needs a FeffCluster")
    edge = _edge(edge)
    if not cluster.elements:
        raise ValueError(f"the cluster about {cluster.label} holds no "
                         "neighbour within "
                         f"{cluster.cluster_radius_ang} Å")
    radius = cluster.cluster_radius_ang
    r_path = _r_path(r_path_ang, radius)
    from .elements import info

    potential: dict[str, int] = {}
    for element in cluster.elements:
        if element not in potential:
            potential[element] = len(potential) + 1
    where = ("MD frame" if cluster.frame_label is None else
             f"MD frame {cluster.frame_label}")
    if cluster.timestep is not None:
        where += f" (timestep {cluster.timestep})"
    lines = [f"* FEFF input written by FACET {FACET_VERSION} on "
             f"{datetime.now():%Y-%m-%d %H:%M}",
             f"* absorber {cluster.label} (atom id {cluster.absorber_id}, "
             f"row {cluster.absorber_row}) in {where}"]
    if source is not None:
        lines.append(f"* source: {_text(source, 'source')}")
    lines.append("* The cluster is every atom within the cluster radius of "
                 "the absorber, cut by distance alone.")
    if cluster.n_potentials > _FEFF8L_MAX_POTENTIALS:
        lines += [f"* This input holds {cluster.n_potentials} unique "
                  f"potentials (ipot 0..{cluster.n_potentials - 1}); the "
                  "FEFF8L shipped with",
                  f"* xraylarch stopped beyond {_FEFF8L_MAX_POTENTIALS} when "
                  "FACET was checked with it (2026-10-07)."]
    lines += [f"* Cluster radius {radius:.2f} A, RPATH {r_path:.2f} A. The "
              "cluster is larger than RPATH on purpose: an atom at its edge",
              "* carries the least complete potential in it.",
              "* A static snapshot of an MD model: no sigma^2 is meant to be "
              "added to its paths; the",
              "* disorder of the model is in the spread over the sampled "
              "clusters."]
    lines += [f"* {text}" for text in extra_comments]
    out = "\n".join(lines) + "\n"
    out += f"\nTITLE {where} {cluster.label}\n\n"
    out += f"EDGE      {edge}\nS02       1.0\n\n"
    out += "CONTROL   1 1 1 1 1 1\n"
    out += "PRINT     0 0 0 0 0 3\n\n"
    out += f"RPATH     {r_path:.2f}\nEXCHANGE  0 0 0\n\n"
    out += f"* ipot 0 is the absorber, {cluster.label}. "
    if cluster.absorber in potential:
        out += (f"ipot {potential[cluster.absorber]} is the other "
                f"{cluster.absorber} atoms,\n*   which need a potential of "
                "their own because ipot 0 is the absorber alone.\n")
    else:
        out += "The others are the scatterers.\n"
    out += "POTENTIALS\n*    ipot   Z  element\n"
    out += f"     {0:<5d} {info(cluster.absorber).z or 0:<3d} " \
           f"{cluster.absorber}\n"
    for element, ipot in potential.items():
        out += f"     {ipot:<5d} {info(element).z or 0:<3d} {element}\n"
    out += "\nATOMS\n*    x          y          z      ipot  label  distance\n"
    out += f"  {0.0:10.5f} {0.0:10.5f} {0.0:10.5f}   0     " \
           f"{cluster.absorber:<4s} 0.00000\n"
    for element, v, d in zip(cluster.elements, cluster.vec_ang,
                             cluster.d_ang):
        out += (f"  {v[0]:10.5f} {v[1]:10.5f} {v[2]:10.5f}   "
                f"{potential[element]}     {element:<4s} {d:.5f}\n")
    out += "END\n"
    path = Path(path)
    path.write_text(out, encoding="utf-8")
    return path


def sample_absorbers(frames: Sequence[int], absorber_rows, *,
                     n_clusters: int, seed: int
                     ) -> tuple[tuple[int, int], ...]:
    """``n_clusters`` (frame, row) pairs drawn without replacement.

    Uniform over every absorber atom of every frame given, from
    ``numpy.random.default_rng(seed)``; returned sorted by frame, then row.
    ``n_clusters`` and ``seed`` have no default. numpy does not promise the
    same draw from one seed across its versions, so the drawn list itself
    (written to ``clusters.csv`` by :func:`export_feff_inputs`) is the record.
    """
    if isinstance(frames, (str, bytes)):
        raise ValueError("frames needs a sequence of frame indices")
    labels = [_whole(k, "a frame index", 0) for k in frames]
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("frames needs distinct frame indices, at least one")
    rows = np.array(absorber_rows, dtype=np.int64).reshape(-1)
    if rows.size == 0 or len(np.unique(rows)) != rows.size or \
            (rows < 0).any():
        raise ValueError("absorber_rows needs distinct rows of 0 or more")
    k = _whole(n_clusters, "n_clusters", 1)
    seed = _whole(seed, "seed", 0)
    total = len(labels) * rows.size
    if k > total:
        raise ValueError(f"{k} clusters asked for, out of {total} absorber "
                         "atoms over the frames given")
    pick = np.random.default_rng(seed).choice(total, size=k, replace=False)
    out = sorted((labels[p // rows.size], int(rows[p % rows.size]))
                 for p in pick.tolist())
    return tuple(out)


@dataclass(frozen=True)
class FeffRequest:
    """What :func:`export_feff_inputs` / :func:`md_exafs` write for FEFF.

    No default for the number of clusters, the seed, the radius or the edge.
    ``overwrite=False`` refuses an ``out_dir`` that already holds a
    ``clusters.csv`` (a draw made earlier, which a new list would leave on
    disk but out of the record) or a drawn cluster's ``feff.inp`` (whose FEFF
    output would no longer match a new input). ``overwrite=True`` replaces
    them, removes the FEFF output beside every ``feff.inp`` it rewrites, and
    notes the cluster directories of an earlier draw left in ``out_dir``.
    """

    out_dir: str | Path
    n_clusters: int
    seed: int
    cluster_radius_ang: float
    edge: str
    r_path_ang: float | None = None
    overwrite: bool = False

    def __post_init__(self) -> None:
        put = object.__setattr__
        put(self, "out_dir", Path(self.out_dir))
        put(self, "n_clusters", _whole(self.n_clusters, "n_clusters", 1))
        put(self, "seed", _whole(self.seed, "seed", 0))
        radius = _positive(self.cluster_radius_ang, "cluster_radius_ang")
        put(self, "cluster_radius_ang", radius)
        put(self, "edge", _edge(self.edge))
        put(self, "r_path_ang", _r_path(self.r_path_ang, radius))
        if not isinstance(self.overwrite, (bool, np.bool_)):
            raise ValueError("overwrite is True or False")


@dataclass(frozen=True)
class FeffClusterRecord:
    """One written cluster: where it came from and where it went."""

    frame: int
    row: int
    atom_id: int
    timestep: int | None
    directory: str
    n_neighbours: int
    nearest_ang: float
    n_potentials: int

    def as_row(self) -> dict:
        return {"frame": self.frame, "row": self.row, "atom id": self.atom_id,
                "timestep": self.timestep, "directory": self.directory,
                "neighbours in cluster": self.n_neighbours,
                "nearest neighbour (Å)": self.nearest_ang,
                "unique potentials": self.n_potentials}


@dataclass(frozen=True, eq=False)
class FeffExport:
    """The FEFF inputs written for configurational averaging.

    ``provenance`` is the header written into ``clusters.csv``; pass it to
    :func:`average_feff_chi` so that the averaged chi(k) carries it too.
    """

    request: FeffRequest
    absorber: str
    clusters: tuple[FeffClusterRecord, ...]
    manifest: Path
    provenance: Provenance
    notes: tuple[str, ...]

    @property
    def directories(self) -> tuple[Path, ...]:
        return tuple(Path(c.directory) for c in self.clusters)

    def as_rows(self) -> list[dict]:
        return [c.as_row() for c in self.clusters]


def _cluster_directory(request: FeffRequest, absorber: str, frame: int,
                       atom_id: int) -> Path:
    return request.out_dir / f"frame{frame:06d}_{absorber}{atom_id}"


def _remove_feff_outputs(directory: Path) -> int:
    """Remove FEFF's output files from one cluster directory; their count."""
    removed = 0
    for pattern in _FEFF_OUTPUTS:
        for item in directory.glob(pattern):
            if item.is_file() and item.name.lower() != "feff.inp":
                item.unlink()
                removed += 1
    return removed


class _FeffWriter:
    """Draws the clusters at the first frame and writes them frame by frame."""

    def __init__(self, request: FeffRequest, absorber: str,
                 frames: list[int], *, source: str | None, method: str):
        if not isinstance(request, FeffRequest):
            raise ValueError("feff needs a FeffRequest")
        self.request = request
        self.absorber = absorber
        self.frames = frames
        self.source = source
        self.method = method
        self.chosen: dict[int, list[int]] | None = None
        self.rows: np.ndarray | None = None
        self.records: list[FeffClusterRecord] = []
        self.half_width_ang: float | None = None
        self.removed: dict[str, int] = {}

    def draw(self, frame: Frame) -> None:
        rows = np.flatnonzero(frame.elements == self.absorber)
        if rows.size == 0:
            raise ValueError(f"the frame holds no {self.absorber}")
        picks = sample_absorbers(self.frames, rows,
                                 n_clusters=self.request.n_clusters,
                                 seed=self.request.seed)
        self.rows = rows
        self.chosen = {}
        for k, row in picks:
            self.chosen.setdefault(k, []).append(row)
        if self.request.overwrite:
            return
        manifest = self.request.out_dir / "clusters.csv"
        if manifest.exists():
            raise ValueError(
                f"{manifest} already exists: out_dir holds clusters drawn "
                "earlier, which a new list would leave on disk but out of "
                "the record. Choose another out_dir, or overwrite=True")
        existing = []
        for k, row in picks:
            target = _cluster_directory(self.request, self.absorber, k,
                                        int(frame.atom_id[row])) / "feff.inp"
            if target.exists():
                existing.append(str(target))
        if existing:
            raise ValueError(
                f"{len(existing)} feff.inp file(s) already exist (first: "
                f"{existing[0]}); their FEFF output would no longer match a "
                "new input. Choose another out_dir, or overwrite=True")

    def wanted(self, k: int) -> list[int]:
        return [] if self.chosen is None else self.chosen.get(k, [])

    def write(self, ap: AbsorberPairs, frame: Frame, k: int) -> None:
        if not np.array_equal(np.flatnonzero(frame.elements == self.absorber),
                              self.rows):
            raise ValueError(f"frame {k} holds its {self.absorber} atoms in "
                             "other rows than the first frame")
        self.half_width_ang = ap.half_width_ang if self.half_width_ang is None \
            else min(self.half_width_ang, ap.half_width_ang)
        n_total = self.request.n_clusters
        for row in self.wanted(k):
            cluster = feff_cluster(
                ap, row, cluster_radius_ang=self.request.cluster_radius_ang,
                frame_label=k)
            directory = _cluster_directory(self.request, self.absorber, k,
                                           cluster.absorber_id)
            directory.mkdir(parents=True, exist_ok=True)
            if self.request.overwrite:
                removed = _remove_feff_outputs(directory)
                if removed:
                    self.removed[str(directory)] = removed
            write_feff_input(
                cluster, directory / "feff.inp", edge=self.request.edge,
                r_path_ang=self.request.r_path_ang, source=self.source,
                extra_comments=(f"one of {n_total} clusters drawn with seed "
                                f"{self.request.seed} for configurational "
                                "averaging",))
            self.records.append(FeffClusterRecord(
                frame=k, row=row, atom_id=cluster.absorber_id,
                timestep=frame.timestep, directory=str(directory),
                n_neighbours=len(cluster.elements),
                nearest_ang=float(cluster.d_ang[0]) if len(cluster.d_ang)
                else float("nan"), n_potentials=cluster.n_potentials))

    def finish(self, trajectory) -> FeffExport:
        from .exporters import write_csv

        request = self.request
        notes = (
            f"{len(self.records)} {self.absorber} clusters of radius "
            f"{request.cluster_radius_ang!r} Å drawn uniformly without "
            f"replacement from {len(self.frames)} frames x "
            f"{self.rows.size} {self.absorber} atoms "
            f"(numpy default_rng, seed {request.seed}); edge "
            f"{request.edge}, RPATH {request.r_path_ang!r} Å",
            "FACET does not run FEFF; average_feff_chi reads the directories "
            "back once FEFF has run in them")
        if self.half_width_ang is not None and \
                request.cluster_radius_ang > self.half_width_ang:
            notes += (
                f"the cluster radius {request.cluster_radius_ang!r} Å exceeds "
                f"half the smallest box width ({self.half_width_ang!r} Å), "
                "so a cluster holds periodic images of atoms already in it, "
                "the absorber's own images included",)
        many = [c for c in self.records
                if c.n_potentials > _FEFF8L_MAX_POTENTIALS]
        if many:
            notes += (
                f"{len(many)} of {len(self.records)} clusters hold more than "
                f"{_FEFF8L_MAX_POTENTIALS} unique potentials (up to "
                f"{max(c.n_potentials for c in many)}); the FEFF8L shipped "
                f"with xraylarch stopped beyond {_FEFF8L_MAX_POTENTIALS} when "
                "FACET was checked with it (2026-10-07), returning exit code "
                "0",)
        if self.removed:
            notes += (
                f"overwrite: FEFF output removed beside "
                f"{len(self.removed)} rewritten feff.inp "
                f"({sum(self.removed.values())} files), since it belonged to "
                "the input replaced",)
        if request.overwrite and request.out_dir.is_dir():
            written = {Path(c.directory).name for c in self.records}
            stray = sorted(
                p.name for p in request.out_dir.glob(
                    f"frame[0-9]*_{self.absorber}[0-9]*")
                if p.is_dir() and p.name not in written)
            if stray:
                notes += (
                    f"{len(stray)} cluster director(ies) of an earlier draw "
                    "stay in out_dir and are not in this clusters.csv: "
                    + ", ".join(stray[:5]) + (" ..." if len(stray) > 5
                                              else ""),)
        provenance = Provenance.from_trajectory(
            trajectory, frames_used=self.frames,
            r_search_ang=request.cluster_radius_ang,
            method_parameters={"n_clusters": request.n_clusters,
                               "seed": request.seed,
                               "cluster_radius_ang": request.cluster_radius_ang,
                               "r_path_ang": request.r_path_ang,
                               "edge": request.edge,
                               "pair search method": self.method},
            notes=notes)
        request.out_dir.mkdir(parents=True, exist_ok=True)
        manifest = request.out_dir / "clusters.csv"
        records = tuple(sorted(self.records, key=lambda c: (c.frame, c.row)))
        write_csv([c.as_row() for c in records], manifest,
                  provenance=provenance.as_lines())
        return FeffExport(request=request, absorber=self.absorber,
                          clusters=records, manifest=manifest,
                          provenance=provenance, notes=notes)


def export_feff_inputs(trajectory, absorber: str, feff: FeffRequest, *,
                       frames: Sequence[int] | None = None,
                       method: str = "auto",
                       progress: Callable[[int, int], None] | None = None,
                       cancelled: Callable[[], bool] | None = None
                       ) -> FeffExport:
    """feff.inp for ``feff.n_clusters`` absorbers drawn across the frames.

    One ``bulk.iter_pairs`` per frame that holds a drawn absorber, to the
    cluster radius with vectors, and only those frames are read beyond the
    first. Each cluster goes to ``out_dir/frameKKKKKK_<element><id>/feff.inp``
    and the list to ``out_dir/clusters.csv``, with a provenance header.
    """
    absorber = validate_symbol(str(absorber))
    indices = _frame_indices(trajectory, frames)
    writer = _FeffWriter(feff, absorber, indices,
                         source=getattr(trajectory, "source_path", None),
                         method=method)
    writer.draw(trajectory.frame(indices[0]))
    todo = [k for k in indices if writer.wanted(k)]
    for done, k in enumerate(todo):
        _tick(progress, cancelled, done, len(todo))
        frame = trajectory.frame(k)
        radius = feff.cluster_radius_ang
        ap = absorber_pairs(frame, iter_pairs(frame, radius,
                                              vectors_within_ang=radius,
                                              method=method), absorber)
        writer.write(ap, frame, k)
    _tick(progress, cancelled, len(todo), len(todo))
    return writer.finish(trajectory)


@dataclass(frozen=True, eq=False)
class AveragedChi:
    """chi(k) averaged over FEFF runs on sampled clusters.

    ``per_cluster`` (n_used, n_k) in the order of ``used``; ``chi_std`` is the
    spread across clusters (ddof = 1). ``missing`` maps each directory left
    out to the reason. ``path_set`` says which paths of each run were
    summed; ``n_paths`` maps each directory used to the number summed, and
    ``n_listed`` to the path files its ``files.dat`` lists. ``provenance``
    is the draw's header (``FeffExport.provenance``) when the caller gave it.
    """

    k_inv_ang: np.ndarray
    chi_mean: np.ndarray
    chi_std: np.ndarray
    per_cluster: np.ndarray
    used: tuple[str, ...]
    missing: Mapping[str, str]
    s02: float
    path_set: str
    n_paths: Mapping[str, int]
    n_listed: Mapping[str, int]
    notes: tuple[str, ...]
    provenance: Provenance | None = None

    def as_rows(self) -> list[dict]:
        return [{"k (1/Å)": float(k), "chi mean": float(m),
                 "chi std across clusters": float(s),
                 "clusters": len(self.used)}
                for k, m, s in zip(self.k_inv_ang, self.chi_mean,
                                   self.chi_std)]

    def cluster_rows(self) -> list[dict]:
        """One row per directory given: used with its path counts, or left
        out with the reason."""
        rows = [{"directory": d, "used": True,
                 "paths summed": self.n_paths[d],
                 "path files listed": self.n_listed[d], "reason left out": ""}
                for d in self.used]
        rows += [{"directory": d, "used": False, "paths summed": None,
                  "path files listed": None, "reason left out": reason}
                 for d, reason in self.missing.items()]
        return rows

    def as_sheets(self) -> dict[str, list[dict]]:
        """Tables for ``exporters.write_xlsx``."""
        sheets = {}
        if self.provenance is not None:
            sheets["provenance"] = self.provenance.as_rows()
        sheets["chi"] = self.as_rows()
        sheets["clusters"] = self.cluster_rows()
        sheets["notes"] = [{"note": note} for note in self.notes]
        return sheets


class _LeftOut(Exception):
    """A FEFF directory left out of the average, with the reason."""


def _path_set(value):
    """'chi.dat', 'all', or a minimum amplitude ratio in per cent."""
    if isinstance(value, str):
        if value not in PATH_SETS:
            raise ValueError(f"path_set {value!r} is not one of {PATH_SETS} "
                             "or a minimum amplitude ratio in per cent")
        return value
    pct = _number(value, "path_set (a minimum amplitude ratio in per cent)")
    if not 0.0 <= pct <= 100.0:
        raise ValueError(f"path_set {pct!r} %: an amplitude ratio between 0 "
                         "and 100 per cent is needed")
    return pct


def _feff_inp_rpath(path: Path) -> float | None:
    for line in path.read_text(encoding="utf-8",
                               errors="replace").splitlines():
        parts = line.split()
        if len(parts) > 1 and parts[0].upper() == "RPATH":
            try:
                return float(parts[1])
            except ValueError:
                return None
    return None


def _chi_dat_list(path: Path):
    """(path indices, (used, total) or None, criterion % or None) from the
    header of FEFF's chi.dat."""
    indices: list[int] = []
    used = None
    criterion = None
    for line in path.read_text(encoding="utf-8",
                               errors="replace").splitlines():
        if not line.startswith("#"):
            break
        match = _CHI_PATH.match(line)
        if match:
            indices.append(int(match.group(1)))
            continue
        match = _CHI_USED.match(line)
        if match:
            used = (int(match.group(1)), int(match.group(2)))
            continue
        match = _CHI_FILTER.search(line)
        if match:
            criterion = float(match.group(1))
    return indices, used, criterion


def _run_paths(directory: Path, k: np.ndarray, path_set):
    """(paths to sum, path files listed, ff2x criterion, whether a feff.inp
    was there to check the output against) of one FEFF run.

    Raises _LeftOut with the reason when the run is incomplete, stale, or
    does not cover the k grid.
    """
    files_dat = directory / "files.dat"
    if not files_dat.is_file():
        raise _LeftOut("no files.dat (FEFF writes it with PRINT 0 0 0 0 0 3, "
                       "which FACET's feff.inp asks for)")
    feff_inp = directory / "feff.inp"
    if feff_inp.is_file():
        if files_dat.stat().st_mtime < feff_inp.stat().st_mtime:
            raise _LeftOut("files.dat is older than feff.inp: FEFF's output "
                           "here predates the input beside it")
        rpath = _feff_inp_rpath(feff_inp)
        match = _RMAX.search(files_dat.read_text(encoding="utf-8",
                                                 errors="replace"))
        if rpath is not None and match is not None and \
                abs(float(match.group(1)) - rpath) > _RMAX_PRINT_TOL_ANG:
            raise _LeftOut(f"files.dat says Rmax = {match.group(1)} Å while "
                           f"feff.inp says RPATH {rpath:g} Å: the output is "
                           "from another input")
    listing = exafs.read_files_dat(files_dat)
    if not listing:
        raise _LeftOut("files.dat lists no path")
    listed = {p.filename.lower() for p in listing}
    on_disk = {p.name.lower() for p in directory.glob("feff[0-9]*.dat")}
    absent = sorted(listed - on_disk)
    if absent:
        raise _LeftOut(f"files.dat lists {len(absent)} path file(s) that are "
                       f"not in the directory (first: {absent[0]})")
    unlisted = sorted(on_disk - listed)
    if unlisted:
        raise _LeftOut(f"{len(unlisted)} path file(s) are not in files.dat "
                       f"(first: {unlisted[0]}): they come from another FEFF "
                       "run")
    paths = exafs.read_feff_directory(directory)
    unreadable = sorted(p.filename for p in paths
                        if p.k is None or p.magnitude is None
                        or p.phase is None or len(p.k) < 2)
    if unreadable:
        raise _LeftOut(f"{len(unreadable)} path file(s) hold no amplitude "
                       f"table (first: {unreadable[0]})")
    criterion = None
    if path_set == "chi.dat":
        chi_dat = directory / "chi.dat"
        if not chi_dat.is_file():
            raise _LeftOut("no chi.dat, whose header lists the paths FEFF's "
                           "own chi(k) sums; path_set='all' or a minimum "
                           "amplitude ratio sums the path files without it")
        if chi_dat.stat().st_mtime < files_dat.stat().st_mtime:
            raise _LeftOut("chi.dat is older than files.dat: its path list "
                           "is from an earlier FEFF run")
        indices, used, criterion = _chi_dat_list(chi_dat)
        if not indices:
            raise _LeftOut("chi.dat lists no path in its header")
        if used is not None and (used[0] != len(indices)
                                 or used[1] != len(listing)):
            raise _LeftOut(f"chi.dat says {used[0]}/{used[1]} paths used, "
                           f"while its header lists {len(indices)} and "
                           f"files.dat {len(listing)}")
        by_index = {p.index: p for p in paths}
        lost = [i for i in indices if i not in by_index]
        if lost:
            raise _LeftOut(f"chi.dat names {len(lost)} path(s) with no path "
                           f"file (first: {lost[0]})")
        chosen = [by_index[i] for i in indices]
    elif path_set == "all":
        chosen = list(paths)
    else:
        chosen = [p for p in paths if np.isfinite(p.amplitude_ratio)
                  and p.amplitude_ratio >= path_set]
        if not chosen:
            raise _LeftOut(f"no path has an amplitude ratio of {path_set:g} % "
                           "or more")
    low = max(float(p.k.min()) for p in chosen)
    high = min(float(p.k.max()) for p in chosen)
    if k[0] < low or k[-1] > high:
        raise _LeftOut(f"FEFF's k range [{low:g}, {high:g}] 1/Å does not "
                       f"cover the grid [{k[0]:g}, {k[-1]:g}]")
    return chosen, len(listing), criterion, feff_inp.is_file()


def average_feff_chi(directories: Sequence[str | Path], *, k_inv_ang,
                     s02: float, path_set: str | float = "chi.dat",
                     provenance: Provenance | None = None) -> AveragedChi:
    """Read FEFF's output in each directory and average chi(k).

    Per directory: the paths of ``path_set``, read with
    ``exafs.read_feff_directory`` and summed by
    ``exafs.chi_from_paths(paths, k=k_inv_ang, s02=s02)`` with no sigma^2
    (the clusters are static snapshots). ``path_set`` is a method choice,
    stated in the result: 'chi.dat' (the default) sums the paths listed in
    the header of FEFF's own ``chi.dat``, the set its ff2x summed, so each
    cluster's chi(k) is FEFF's; 'all' sums every path file ``files.dat``
    lists; a number keeps the paths whose curved-wave amplitude ratio in
    ``files.dat`` is at least that many per cent. ``k_inv_ang`` (a grid
    above 0) and ``s02`` have no default.

    A directory is left out, listed in ``missing`` with the reason and
    noted, when it holds no ``files.dat``; when ``files.dat`` is older than
    the ``feff.inp`` beside it or its Rmax is not that file's RPATH (output
    of another input); when a path file it lists is absent or holds no
    amplitude table, or a path file is not in its list (an incomplete or
    mixed run); when 'chi.dat' is asked for and ``chi.dat`` is absent, older
    than ``files.dat`` or inconsistent with it; and when FEFF's k range does
    not cover the grid (``numpy.interp`` would hold the end values). A
    directory with no ``feff.inp`` beside its output is used, and noted as
    not checked against an input. A directory given twice is refused: it
    would count twice in the mean and narrow the spread.
    ``provenance`` (``FeffExport.provenance``) is attached for the export.
    """
    try:
        k = np.array(k_inv_ang, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("k_inv_ang needs numbers") from None
    if k.ndim != 1 or k.size < 2 or not np.isfinite(k).all() or \
            not (np.diff(k) > 0).all() or k[0] <= 0.0:
        raise ValueError("k_inv_ang needs a strictly increasing 1-D grid "
                         "above 0, two points or more")
    s02 = _positive(s02, "s02")
    chosen_set = _path_set(path_set)
    if provenance is not None and not isinstance(provenance, Provenance):
        raise ValueError("provenance needs an md_stats.Provenance or None")
    if isinstance(directories, (str, Path)):
        directories = [directories]
    directories = [Path(entry) for entry in directories]
    seen: dict[str, str] = {}
    for directory in directories:
        same = os.path.normcase(str(directory.resolve()))
        if same in seen:
            raise ValueError(
                f"the directory {directory} is given twice (also as "
                f"{seen[same]}): one FEFF run would count twice in the mean "
                "and narrow the spread across clusters")
        seen[same] = str(directory)
    rows, used, missing = [], [], {}
    n_paths: dict[str, int] = {}
    n_listed: dict[str, int] = {}
    criteria = set()
    unchecked = []
    for directory in directories:
        key = str(directory)
        if not directory.is_dir():
            missing[key] = "directory not found"
            continue
        try:
            paths, listed, criterion, checked = _run_paths(directory, k,
                                                           chosen_set)
            chi = exafs.chi_from_paths(paths, k=k, s02=s02)
            if len(chi.per_path) != len(paths):
                raise _LeftOut(f"{len(paths) - len(chi.per_path)} of the "
                               f"{len(paths)} paths have no positive "
                               "effective length")
        except _LeftOut as reason:
            missing[key] = str(reason)
            continue
        if criterion is not None:
            criteria.add(criterion)
        rows.append(np.asarray(chi.chi, dtype=np.float64))
        used.append(key)
        n_paths[key] = len(paths)
        n_listed[key] = listed
        if not checked:
            unchecked.append(key)
    if not rows:
        raise ValueError(f"no directory holds FEFF output to read ("
                         f"{len(missing)} given: "
                         + "; ".join(f"{d}: {r}" for d, r in
                                     list(missing.items())[:3]) + ")")
    table = np.stack(rows)
    mean, spread = mean_and_spread(table)
    if chosen_set == "chi.dat":
        described = ("the paths FEFF's own ff2x summed into chi.dat (listed "
                     "in its header)")
    elif chosen_set == "all":
        described = "every path file FEFF wrote (listed in files.dat)"
    else:
        described = (f"the paths of curved-wave amplitude ratio "
                     f"{chosen_set:g} % or more (files.dat)")
    counts = list(n_paths.values())
    listed_counts = list(n_listed.values())
    notes = [
        f"chi(k) of each cluster summed over {described}: {min(counts)} to "
        f"{max(counts)} paths per cluster, of {min(listed_counts)} to "
        f"{max(listed_counts)} path files FEFF wrote (per cluster in n_paths "
        "and n_listed)",
        "the amplitudes, phases, mean free paths and degeneracies are FEFF's "
        "(exafs.chi_from_paths); nothing is fitted to a measurement"]
    if criteria:
        notes.append("the curved-wave amplitude ratio filter in chi.dat: "
                     + ", ".join(f"{c:g} %" for c in sorted(criteria)))
    notes.append(f"chi(k) averaged over {len(used)} clusters at "
                 f"S0^2 = {s02!r}; std is the spread across clusters "
                 "(ddof = 1), not a standard error")
    notes.append("sigma^2 = 0 on every path: each cluster is a static "
                 "snapshot, and the disorder of the model is in the spread "
                 "over the clusters")
    if missing:
        notes.append(f"{len(missing)} director(ies) left out: "
                     + "; ".join(f"{d}: {r}" for d, r in missing.items()))
    if unchecked:
        notes.append(
            f"{len(unchecked)} of {len(used)} directories used hold no "
            "feff.inp beside the FEFF output, so the output was not checked "
            "against the input it came from (files.dat older than feff.inp, "
            "Rmax against RPATH): " + ", ".join(unchecked[:5])
            + (" ..." if len(unchecked) > 5 else ""))
    if len(used) == 1:
        notes.append("one cluster: the spread across clusters is undefined "
                     "(NaN)")
    if provenance is None:
        notes.append("no provenance was given, so the trajectory, frames and "
                     "draw behind these clusters are not recorded with this "
                     "average (FeffExport.provenance holds them)")
    frozen = md_stats._frozen
    return AveragedChi(k_inv_ang=frozen(k), chi_mean=frozen(mean),
                       chi_std=frozen(spread), per_cluster=frozen(table),
                       used=tuple(used), missing=dict(missing), s02=s02,
                       path_set=described, n_paths=n_paths,
                       n_listed=n_listed, notes=tuple(notes),
                       provenance=provenance)


# ---------------------------------------------------------------------------
# a trajectory, end to end
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class MdExafs:
    """Absorber-centred g(r), shells and their cumulants over frames.

    ``g`` and ``n_cum`` map each neighbour element to an ``md_stats.Series``
    on r_ang. ``limits`` holds each neighbour's first-shell limit (auto or
    user). ``shells`` the averaged cumulants of every shell (first shells,
    then the user's, in the order given). ``feff`` the FEFF inputs written,
    if asked for. ``searches_per_frame`` is 1 when every shell limit was
    known before the first frame, 2 when the limits came from the
    frame-averaged g(r).
    """

    absorber: str
    g: Mapping[str, Series]
    n_cum: Mapping[str, Series]
    limits: Mapping[str, ShellLimit]
    shells: tuple[AveragedShell, ...]
    feff: FeffExport | None
    provenance: Provenance
    searches_per_frame: int
    notes: tuple[str, ...]

    def as_sheets(self) -> dict[str, list[dict]]:
        sheets = {"provenance": self.provenance.as_rows()}
        for element, series in self.g.items():
            sheets[f"g {self.absorber}-{element}"] = series.as_rows()
            sheets[f"N {self.absorber}-{element}"] = \
                self.n_cum[element].as_rows()
        sheets["shell limits"] = [
            {"absorber": x.absorber, "neighbour": x.neighbour,
             "limit (Å)": x.r_ang, "g at limit": x.g_at_limit,
             "first-shell maximum (Å)": x.r_peak_ang,
             "g at first-shell maximum": x.g_peak,
             "uncorrelated level": x.level,
             "floor from (Å)": None if x.floor_r_ang is None
             else x.floor_r_ang[0],
             "floor to (Å)": None if x.floor_r_ang is None
             else x.floor_r_ang[1],
             "g standard error at limit": x.g_std_error,
             "floor depth (standard errors)": x.depth_std_errors,
             "source": x.source, "method": x.method,
             "no limit because": x.reason,
             "notes": " | ".join(x.notes)} for x in self.limits.values()]
        sheets["shells"] = [row for shell in self.shells
                            for row in shell.as_rows()]
        if self.feff is not None:
            sheets["feff clusters"] = self.feff.as_rows()
        sheets["notes"] = [{"note": n} for n in self.notes]
        return sheets


def md_exafs(trajectory, absorber: str, *, r_max_ang: float, dr_ang: float,
             minimum: glass.MinimumMethod | None,
             frames: Sequence[int] | None = None,
             neighbours: Sequence[str] | None = None,
             first_shell_limits_ang: Mapping[str, float] | None = None,
             shells: Sequence[tuple[str, float, float]] = (),
             weighting: str = "none", feff: FeffRequest | None = None,
             method: str = "auto",
             progress: Callable[[int, int], None] | None = None,
             cancelled: Callable[[], bool] | None = None) -> MdExafs:
    """Absorber-centred partial g(r), first shells, cumulants, FEFF inputs.

    Pass 1, per frame: one ``bulk.iter_pairs`` to the last point of the g(r)
    grid plus ``dr_ang`` (:func:`absorber_rdf`), or to the outermost shell
    limit when that lies further and every shell limit is known in advance
    (and to the FEFF cluster radius, with vectors to it, in a frame that
    holds a drawn cluster only); the partial g(r) and N(r) of each
    neighbour; the FEFF clusters drawn in that frame; and, when every limit
    is known, the cumulants of every shell. The
    first-shell limit of each neighbour is ``first_shell_limits_ang[neighbour]``
    when given, else :func:`first_minimum` of the frame-averaged g(r) under
    ``minimum`` (a ``glass.MinimumMethod``; None only when every neighbour's
    limit is given), with the uncorrelated level and the counting errors of
    the averaged g taken from the model's atom counts, the mean volume and
    the number of frames. When some limit came from the averaged g(r), pass 2
    reads each frame again and searches only to the outermost shell limit,
    and every shell is measured there. ``shells`` adds (neighbour, r_lo_ang,
    r_hi_ang) shells of the user's; a shell given twice, or equal to a given
    first shell, is refused. ``r_max_ang``, ``dr_ang`` and ``minimum`` have
    no default; a ``minimum`` given when every limit is given too is not
    used, and a note says so. r_max, every given limit and every shell's
    outer edge are checked against half the first frame's smallest box
    width before any pair search.
    """
    absorber = validate_symbol(str(absorber))
    indices = _frame_indices(trajectory, frames)
    r_max = _positive(r_max_ang, "r_max_ang")
    _, r_grid_need = _rdf_points(r_max, dr_ang)
    weighting = _weighting(weighting)
    if minimum is not None and not isinstance(minimum, glass.MinimumMethod):
        raise ValueError("minimum needs a glass.MinimumMethod (rule, "
                         "smooth_sigma_ang, flat_rule, margin_std_errors), or "
                         "None when every first-shell limit is given")
    given = {}
    for element, limit in (first_shell_limits_ang or {}).items():
        given[validate_symbol(str(element))] = _positive(
            limit, f"the first-shell limit of {element}")
    extra = []
    if isinstance(shells, (str, bytes)):
        raise ValueError("shells needs a sequence of (neighbour, r_lo_ang, "
                         "r_hi_ang)")
    for item in shells:
        if isinstance(item, (str, bytes)) or len(item) != 3:
            raise ValueError("each shell needs (neighbour, r_lo_ang, r_hi_ang)")
        element = validate_symbol(str(item[0]))
        low, high = _number(item[1], "r_lo_ang"), _positive(item[2],
                                                            "r_hi_ang")
        if low < 0 or low >= high:
            raise ValueError(f"the shell {element} [{low}, {high}] Å needs "
                             "0 <= r_lo < r_hi")
        if (element, low, high) in extra:
            raise ValueError(f"the shell {element} ({low:g}, {high:g}] Å is "
                             "given twice")
        if low == 0.0 and given.get(element) == high:
            raise ValueError(f"the shell {element} (0, {high:g}] Å is the "
                             f"first shell of the limit given for {element}; "
                             "it is reported once, as the first shell")
        extra.append((element, low, high))
    writer = None if feff is None else _FeffWriter(
        feff, absorber, indices,
        source=getattr(trajectory, "source_path", None), method=method)

    first = trajectory.frame(indices[0])
    present = sorted(set(str(s) for s in np.unique(first.elements)))
    if neighbours is None:
        chosen = present
    else:
        if isinstance(neighbours, (str, bytes)):
            raise ValueError("neighbours needs a collection of symbols")
        chosen = [validate_symbol(str(s)) for s in neighbours]
    unknown = sorted((set(chosen) | set(given) | {e for e, _, _ in extra}
                      | {absorber}) - set(present))
    if unknown:
        raise ValueError(f"the model holds no {', '.join(unknown)}")
    unused = sorted(set(given) - set(chosen))
    if unused:
        raise ValueError(f"a first-shell limit is given for "
                         f"{', '.join(unused)}, which is not among the "
                         f"neighbours ({', '.join(chosen)})")
    known = all(element in given for element in chosen)
    if not known and minimum is None:
        lacking = [e for e in chosen if e not in given]
        raise ValueError(f"no first-shell limit is given for "
                         f"{', '.join(lacking)} and minimum is None: a "
                         "glass.MinimumMethod is needed to locate it in the "
                         "frame-averaged g(r)")
    # before any pair search: every radius against the first frame's box
    half = float(first.perpendicular_widths_ang.min() / 2.0)
    _within_half_width(r_max, half, "r_max_ang")
    for element, limit in given.items():
        _within_half_width(limit, half, f"the first-shell limit of {element}")
    for element, _, high in extra:
        _within_half_width(high, half, f"the outer edge of the {element} "
                                       "shell")
    if writer is not None:
        writer.draw(first)
    # pass 1 measures the shells only when every limit is known; otherwise
    # pass 2 measures them all, and pass 1 needs the g(r) grid alone
    r_known = max([given.get(e, 0.0) for e in chosen]
                  + [h for _, _, h in extra] + [0.0]) if known else 0.0
    radius = feff.cluster_radius_ang if feff is not None else 0.0
    r_base = max(r_grid_need, r_known)
    passes = 1 if known else 2
    total = len(indices) * passes

    g_rows = {e: [] for e in chosen}
    n_rows = {e: [] for e in chosen}
    grid = None
    # shells kept in a list, by position: two specs that happen to be equal
    # (a user shell and an automatic first shell) stay two shells
    specs = ([(e, 0.0, given[e], "user") for e in chosen if e in given]
             + [(e, lo, hi, "user") for e, lo, hi in extra]) if known else []
    per_shell: list[list[ShellCumulants]] = [[] for _ in specs]
    radii_used = []
    volumes = []
    counts: dict[str, tuple[int, int]] = {}
    for done, k in enumerate(indices):
        _tick(progress, cancelled, done, total)
        frame = first if k == indices[0] else trajectory.frame(k)
        holds_cluster = writer is not None and bool(writer.wanted(k))
        r_frame = max(r_base, radius) if holds_cluster else r_base
        radii_used.append(r_frame)
        ap = absorber_pairs(frame, iter_pairs(
            frame, r_frame, vectors_within_ang=radius if holds_cluster
            else 0.0, method=method), absorber)
        rdfs = absorber_rdf(ap, r_max_ang=r_max, dr_ang=dr_ang,
                            neighbours=chosen)
        volumes.append(ap.volume_ang3)
        for element, rdf in rdfs.items():
            g_rows[element].append(rdf.g)
            n_rows[element].append(rdf.n_cum)
            grid = rdf.r_ang
            pair_counts = (rdf.n_absorbers, rdf.n_neighbours)
            if counts.setdefault(element, pair_counts) != pair_counts:
                raise ValueError(f"frame {k} holds other numbers of "
                                 f"{absorber} or {element} atoms than the "
                                 "first frame")
        for c, spec in enumerate(specs):
            per_shell[c].append(shell_cumulants(
                ap, spec[0], r_lo_ang=spec[1], r_hi_ang=spec[2],
                weighting=weighting))
        if holds_cluster:
            writer.write(ap, frame, k)
    del first

    g_series, n_series, limits = {}, {}, {}
    for element in chosen:
        name = f"{absorber}-{element}"
        g_series[element] = Series.from_frames(
            grid, g_rows[element], name=f"g {name}(r)", axis_name="r_ang",
            axis_unit="Å", value_unit="1", frames=indices)
        n_series[element] = Series.from_frames(
            grid, n_rows[element], name=f"N {name}(r)", axis_name="r_ang",
            axis_unit="Å", value_unit="1", frames=indices,
            notes=(f"the exact number of {element} atoms within r of a "
                   f"{absorber} atom, per {absorber} atom",))
        if element in given:
            limits[element] = ShellLimit(
                absorber, element, given[element], None, None, "user",
                "first-shell limit given by the user")
            continue
        # glass's level and counting errors, as its analysis passes them for
        # a frame-averaged g(r) (the mean volume of the frames used)
        n_a, n_b = counts[element]
        same = element == absorber
        if same and n_a < 2:
            limits[element] = ShellLimit(
                absorber, element, None, None, None, "auto",
                minimum.describe(), reason=f"the model holds one {absorber} "
                f"atom, so it has no {absorber}-{absorber} pair")
            continue
        per_unit = glass.pair_counts_per_unit_g(
            grid, float(dr_ang), n_a, n_b, math.fsum(volumes) / len(volumes),
            same_element=same, n_frames=len(indices))
        limits[element] = first_minimum(
            grid, g_series[element].mean, method=minimum, absorber=absorber,
            neighbour=element, level=(n_a - 1) / n_a if same else 1.0,
            pairs_per_unit_g=per_unit)
    notes = []
    for element, limit in limits.items():
        if limit.r_ang is None:
            notes.append(f"{absorber}-{element}: no first-shell limit "
                         f"({limit.reason}); no first shell is reported for "
                         "it")
        notes.extend(limit.notes)
    r_first = max(radii_used)
    more = sum(1 for r in radii_used if r > r_base)
    searched = (f"to {r_base!r} Å" + (f" ({more} frame(s) holding a drawn "
                                       f"FEFF cluster to {radius!r} Å)"
                                       if more else ""))
    r_second = None
    if not known:
        auto = "first minimum of the frame-averaged g(r)"
        specs = ([(e, 0.0, limits[e].r_ang, auto) for e in chosen
                  if limits[e].r_ang is not None]
                 + [(e, lo, hi, "user") for e, lo, hi in extra])
        per_shell = [[] for _ in specs]
        for e, lo, hi in extra:
            if lo == 0.0 and limits[e].r_ang == hi and e not in given:
                notes.append(f"the user shell {e} (0, {hi:g}] Å equals the "
                             f"automatic first shell of {e}; both are listed")
        if specs:
            r_second = max(s[2] for s in specs)
            for done, k in enumerate(indices):
                _tick(progress, cancelled, len(indices) + done, total)
                frame = trajectory.frame(k)
                ap = absorber_pairs(frame, iter_pairs(
                    frame, r_second, vectors_within_ang=0.0, method=method),
                    absorber)
                for c, spec in enumerate(specs):
                    per_shell[c].append(shell_cumulants(
                        ap, spec[0], r_lo_ang=spec[1], r_hi_ang=spec[2],
                        weighting=weighting))
            notes.append(
                "the first-shell limits came from the frame-averaged g(r), "
                "known only after every frame was read, so each frame was "
                f"read twice: pass 1 searched {searched} for g(r), pass 2 to "
                f"{r_second!r} Å for the shell distances")
        else:
            passes = 1
            notes.append(f"no first-shell limit was found and no shell was "
                         f"given, so no shell was measured: one pair search "
                         f"per frame, {searched}")
    else:
        notes.append(f"every shell limit was known in advance: one pair "
                     f"search per frame, {searched}")
        if minimum is not None:
            notes.append(f"a first-minimum method was given ({minimum.rule} "
                         "rule) but every first-shell limit was given too, so "
                         "no limit was located with it and it is not in the "
                         "provenance")
    _tick(progress, cancelled, total, total)
    averaged = [average_shells(items, frames=indices, limit_source=spec[3])
                for spec, items in zip(specs, per_shell)]
    export = None if writer is None else writer.finish(trajectory)
    cutoffs = {(absorber, e): limits[e].r_ang for e in chosen
               if limits[e].r_ang is not None}
    sources = {(absorber, e): (limits[e].source if limits[e].source == "user"
                               else limits[e].method)
               for e in chosen if limits[e].r_ang is not None}
    method_parameters = {"r_max_ang": r_max, "dr_ang": float(dr_ang),
                         "weighting": weighting,
                         "user shells": tuple(f"{e} ({lo:g}, {hi:g}] Å"
                                              for e, lo, hi in extra),
                         "pair search method": method,
                         "pass 1 search radius (Å)": r_base}
    if minimum is not None and not known:
        method_parameters.update({
            "first minimum": minimum.describe(),
            "first minimum rule": minimum.rule,
            "first minimum smooth_sigma_ang": minimum.smooth_sigma_ang,
            "first minimum flat_rule": minimum.flat_rule,
            "first minimum margin_std_errors": minimum.margin_std_errors})
    if r_second is not None:
        method_parameters["pass 2 search radius (Å)"] = r_second
    if feff is not None:
        method_parameters.update({"feff n_clusters": feff.n_clusters,
                                  "feff seed": feff.seed,
                                  "feff cluster_radius_ang":
                                      feff.cluster_radius_ang,
                                  "feff edge": feff.edge})
    provenance = Provenance.from_trajectory(
        trajectory, frames_used=indices, r_search_ang=r_first,
        cutoffs_ang=cutoffs, cutoff_sources=sources,
        method_parameters=method_parameters, notes=tuple(notes))
    return MdExafs(absorber=absorber, g=g_series, n_cum=n_series,
                   limits=limits, shells=tuple(averaged), feff=export,
                   provenance=provenance, searches_per_frame=passes,
                   notes=tuple(notes))
