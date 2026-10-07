"""The bond-valence model.

A bond valence is  v = exp((R0 - d) / b).  The sum of v over a cation's
contacts should recover its formal charge, and that is the only check the model
offers -- but it is enough to turn "how far do I look?" from a matter of taste
into a matter of bond strength.

Why FACET cuts by valence rather than by distance
-------------------------------------------------
A distance cutoff chosen for oxygen is the wrong cutoff for iodine. Cutting at
a fixed *valence* instead applies the same minimum bond strength to every anion:

    v > V_BOND  is a bond          d < R0 - b*ln(V_BOND)
    v > V_LIST  is worth listing   d < R0 - b*ln(V_LIST)

With the defaults below and b = 0.37 that is d < R0 + 0.958 and d < R0 + 1.447.
For Bi-O it reproduces the conventional 3.05 / 3.54 A almost exactly; for Bi-I
it becomes 3.72 / 4.21 A, which no oxygen-derived convention would have given.

Parameter provenance
--------------------
Every parameter FACET returns says where it came from. A *fitted* parameter was
refined against experimental structures and is quoted with its citation. An
*estimated* parameter is computed from O'Keeffe & Brese's electronegativity
expression, covers any pair of 75 elements, and is worth roughly half the
accuracy of a fitted one. The distinction is carried through every calculation
and is shown in the interface and in exported reports -- it is never silently
dropped.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from . import elements

# --- thresholds --------------------------------------------------------------
# Defaults, not constants. The whole point of the application is that the user
# moves these and watches what happens.
V_BOND_DEFAULT = 0.075   # v.u. -- above this a contact is counted as a bond
V_LIST_DEFAULT = 0.02    # v.u. -- above this a contact is tabulated
D_MIN_CONTACT = 1.60     # A -- below this a "contact" is a split position

B_DEFAULT = 0.37         # A -- Brese & O'Keeffe's universal b


@dataclass(frozen=True)
class BVParam:
    """One cation-anion bond-valence parameter pair, with its provenance."""

    cation: str
    cation_ox: int
    anion: str
    anion_ox: int
    r0: float
    b: float
    source: str
    fitted: bool

    @property
    def label(self) -> str:
        c = f"{self.cation}{self.cation_ox:+d}".replace("+", "+") if self.cation_ox else self.cation
        return f"{c}-{self.anion}"

    def valence(self, d):
        """Bond valence at distance d (angstrom). Accepts scalars or arrays."""
        return np.exp((self.r0 - np.asarray(d, float)) / self.b)

    def distance_for(self, v: float) -> float:
        """The distance at which the bond valence falls to v.

        A valence of zero is reached only at infinite distance, so it is refused
        rather than allowed to surface as ``math domain error`` from inside a
        logarithm -- which is what a threshold of 0 used to produce, three calls
        deep and with nothing in the message about thresholds.
        """
        v = float(v)
        if v <= 0.0:
            raise ValueError(
                f"a bond valence of {v:g} is reached only at infinite "
                "distance; the threshold must be greater than zero")
        return self.r0 - self.b * math.log(v)


# ---------------------------------------------------------------------------
# Fitted parameters
# ---------------------------------------------------------------------------
# Brese & O'Keeffe, Acta Cryst. B47 (1991) 192, Table 2. b = 0.37 throughout.
#
# What Table 2 above does not cover. Table 2 is oxides, fluorides and chlorides
# only, so the other anions are entered by hand from Table 3 of the same paper
# -- the bismuth set among them, audited against Sleight's independent
# compilation (Physica C 514 (2015) 152) during the 2026-08-31 Na-Bi-O work and
# reproduced to 0.001 A. Entries here win over Table 2, so an individual pair
# can still be corrected or audited. Anything absent from both falls through to
# the estimator, which is a better outcome than a mistyped "fitted" value.
#
# `tests/test_bv.py` cross-checks every entry here against the estimator and
# fails on any that disagree by more than 0.20 A, which catches transcription
# errors without requiring the estimator to be accurate.
_BO1991 = "Brese & O'Keeffe, Acta Cryst. B47 (1991) 192, Table 2"

# ---------------------------------------------------------------------------
# Brese & O'Keeffe (1991) Table 2, in full
# ---------------------------------------------------------------------------
# "Recommended bond-valence parameters for oxides, fluorides and chlorides":
# 109 rows -- one per cation and oxidation state -- each with R0 for O, F and
# Cl, b = 0.37 throughout: 327 values, 48 of them printed in italics. The
# paper's text (p. 194) speaks of "the 330 entries in Table 2". The table is set
# as two bands of 55 row slots, and 2 x 55 x 3 = 330 counts the empty last slot
# of the right-hand band, after Zr(IV); the page prints 327 numbers and no
# blank cell.
#
# FACET used to carry 60 pairs entered by hand, of which 47 were oxides and six
# were fluorides -- so a fluoride structure nearly always fell through to the
# estimator. Measured over the pairs where both a fitted value and an estimate
# exist, the estimator is out by 0.05 A rms and by as much as 0.21 A, which is
# a factor of 0.57 to 1.48 on every bond valence. For fluorides it is high by
# 0.038 A on average, so it inflates them systematically. Cryolite's Al came
# out at 3.51 v.u. against +3 for exactly that reason: R0 estimated at 1.609
# where this table gives 1.545.
#
# Transcribed from the paper by span position rather than by OCR token order --
# its text layer spells the decimal point as any of . - " ' and the Roman
# numeral oxidation states as whatever letters they resemble -- and checked
# four ways: against the 53 pairs FACET had already entered by hand, against
# the paper's own relation R(F) = 0.021 + 0.940 R(O), against the elements and
# oxidation states being real, and by reading all four bands of the rendered
# page. tests/test_bv_table2.py re-derives it from the paper when it is there.
#
# Double-entry visual audit, 2026-10-06. Two transcriptions made blind and
# independently from rendered images of page 4 (zoom 8 and 14), compared value
# by value, agree on all 109 rows, 327 values and 48 italic marks. The rows
# where they differed from this file were rendered again (zoom 10 to 24) and
# read. Before the audit this file held 108 rows, 324 values, 47 italic marks:
#   * Ti(III) 1.791 / 1.723 / 2.17 was absent, and Ti(IV) 1.815 / 1.76 / 2.19
#     was keyed ("Tl", 4). The text layer spells both Ti and Tl as "Ti" or "TI",
#     and the paper has no Tl(IV) row. Every Ti lookup fell through to the
#     estimator.
#   * H(I) 0.95 / 0.92 / 1.28 was keyed ("H", 3), after the text layer's "H m"
#     for the superscript I. Every H(+1) lookup fell through to the estimator.
#   * Italic marks: Br(VII)-Cl, Ti(III)-Cl and Ti(IV)-Cl were absent;
#     Tl(III)-Cl (printed upright) and ("Tl", 4, "Cl") were present.
# No printed number changed. These were row labels and marks, not values taken
# from another source, so none of them is recorded in SUPERSEDED below. The
# page test now compares the printed rows one by one, in the paper's order, so
# a dropped or relabelled row fails it.

_TABLE2: dict[tuple[str, int], dict[str, float]] = {
    ("Ac", 3): {"O": 2.24, "F": 2.13, "Cl": 2.63},
    ("Ag", 1): {"O": 1.805, "F": 1.8, "Cl": 2.09},
    ("Al", 3): {"O": 1.651, "F": 1.545, "Cl": 2.03},
    ("Am", 3): {"O": 2.11, "F": 2, "Cl": 2.48},
    ("As", 3): {"O": 1.789, "F": 1.7, "Cl": 2.16},
    ("As", 5): {"O": 1.767, "F": 1.62, "Cl": 2.14},
    ("Au", 3): {"O": 1.833, "F": 1.81, "Cl": 2.17},
    ("B", 3): {"O": 1.371, "F": 1.31, "Cl": 1.74},
    ("Ba", 2): {"O": 2.29, "F": 2.19, "Cl": 2.69},
    ("Be", 2): {"O": 1.381, "F": 1.28, "Cl": 1.76},
    ("Bi", 3): {"O": 2.09, "F": 1.99, "Cl": 2.48},
    ("Bi", 5): {"O": 2.06, "F": 1.97, "Cl": 2.44},
    ("Bk", 3): {"O": 2.08, "F": 1.96, "Cl": 2.46},
    ("Br", 7): {"O": 1.81, "F": 1.72, "Cl": 2.19},
    ("C", 4): {"O": 1.39, "F": 1.32, "Cl": 1.76},
    ("Ca", 2): {"O": 1.967, "F": 1.842, "Cl": 2.37},
    ("Cd", 2): {"O": 1.904, "F": 1.811, "Cl": 2.23},
    ("Ce", 3): {"O": 2.151, "F": 2.036, "Cl": 2.52},
    ("Ce", 4): {"O": 2.028, "F": 1.995, "Cl": 2.41},
    ("Cf", 3): {"O": 2.07, "F": 1.95, "Cl": 2.45},
    ("Cl", 7): {"O": 1.632, "F": 1.55, "Cl": 2},
    ("Cm", 3): {"O": 2.23, "F": 2.12, "Cl": 2.62},
    ("Co", 2): {"O": 1.692, "F": 1.64, "Cl": 2.01},
    ("Co", 3): {"O": 1.7, "F": 1.62, "Cl": 2.05},
    ("Cr", 2): {"O": 1.73, "F": 1.67, "Cl": 2.09},
    ("Cr", 3): {"O": 1.724, "F": 1.64, "Cl": 2.08},
    ("Cr", 6): {"O": 1.794, "F": 1.74, "Cl": 2.12},
    ("Cs", 1): {"O": 2.42, "F": 2.33, "Cl": 2.79},
    ("Cu", 1): {"O": 1.593, "F": 1.6, "Cl": 1.85},
    ("Cu", 2): {"O": 1.679, "F": 1.6, "Cl": 2},
    ("Dy", 3): {"O": 2.036, "F": 1.922, "Cl": 2.41},
    ("Er", 3): {"O": 2.01, "F": 1.906, "Cl": 2.39},
    ("Eu", 2): {"O": 2.147, "F": 2.04, "Cl": 2.53},
    ("Eu", 3): {"O": 2.076, "F": 1.961, "Cl": 2.455},
    ("Fe", 2): {"O": 1.734, "F": 1.65, "Cl": 2.06},
    ("Fe", 3): {"O": 1.759, "F": 1.67, "Cl": 2.09},
    ("Ga", 3): {"O": 1.73, "F": 1.62, "Cl": 2.07},
    ("Gd", 3): {"O": 2.065, "F": 1.95, "Cl": 2.445},
    ("Ge", 4): {"O": 1.748, "F": 1.66, "Cl": 2.14},
    ("H", 1): {"O": 0.95, "F": 0.92, "Cl": 1.28},
    ("Hf", 4): {"O": 1.923, "F": 1.85, "Cl": 2.3},
    ("Hg", 1): {"O": 1.9, "F": 1.81, "Cl": 2.28},
    ("Hg", 2): {"O": 1.93, "F": 1.9, "Cl": 2.25},
    ("Ho", 3): {"O": 2.023, "F": 1.908, "Cl": 2.401},
    ("I", 5): {"O": 2, "F": 1.9, "Cl": 2.38},
    ("I", 7): {"O": 1.93, "F": 1.83, "Cl": 2.31},
    ("In", 3): {"O": 1.902, "F": 1.79, "Cl": 2.28},
    ("Ir", 5): {"O": 1.916, "F": 1.82, "Cl": 2.3},
    ("K", 1): {"O": 2.13, "F": 1.99, "Cl": 2.52},
    ("La", 3): {"O": 2.172, "F": 2.057, "Cl": 2.545},
    ("Li", 1): {"O": 1.466, "F": 1.36, "Cl": 1.91},
    ("Lu", 3): {"O": 1.971, "F": 1.876, "Cl": 2.361},
    ("Mg", 2): {"O": 1.693, "F": 1.581, "Cl": 2.08},
    ("Mn", 2): {"O": 1.79, "F": 1.698, "Cl": 2.13},
    ("Mn", 3): {"O": 1.76, "F": 1.66, "Cl": 2.14},
    ("Mn", 4): {"O": 1.753, "F": 1.71, "Cl": 2.13},
    ("Mn", 7): {"O": 1.79, "F": 1.72, "Cl": 2.17},
    ("Mo", 6): {"O": 1.907, "F": 1.81, "Cl": 2.28},
    ("N", 3): {"O": 1.361, "F": 1.37, "Cl": 1.75},
    ("N", 5): {"O": 1.432, "F": 1.36, "Cl": 1.8},
    ("Na", 1): {"O": 1.8, "F": 1.677, "Cl": 2.15},
    ("Nb", 5): {"O": 1.911, "F": 1.87, "Cl": 2.27},
    ("Nd", 3): {"O": 2.117, "F": 2.008, "Cl": 2.492},
    ("Ni", 2): {"O": 1.654, "F": 1.599, "Cl": 2.02},
    ("Os", 4): {"O": 1.811, "F": 1.72, "Cl": 2.19},
    ("P", 5): {"O": 1.604, "F": 1.521, "Cl": 1.99},
    ("Pb", 2): {"O": 2.112, "F": 2.03, "Cl": 2.53},
    ("Pb", 4): {"O": 2.042, "F": 1.94, "Cl": 2.43},
    ("Pd", 2): {"O": 1.792, "F": 1.74, "Cl": 2.05},
    ("Pr", 3): {"O": 2.135, "F": 2.022, "Cl": 2.5},
    ("Pt", 2): {"O": 1.768, "F": 1.68, "Cl": 2.05},
    ("Pt", 4): {"O": 1.879, "F": 1.759, "Cl": 2.17},
    ("Pu", 3): {"O": 2.11, "F": 2, "Cl": 2.48},
    ("Rb", 1): {"O": 2.26, "F": 2.16, "Cl": 2.65},
    ("Re", 7): {"O": 1.97, "F": 1.86, "Cl": 2.23},
    ("Rh", 3): {"O": 1.791, "F": 1.71, "Cl": 2.17},
    ("Ru", 4): {"O": 1.834, "F": 1.74, "Cl": 2.21},
    ("S", 4): {"O": 1.644, "F": 1.6, "Cl": 2.02},
    ("S", 6): {"O": 1.624, "F": 1.56, "Cl": 2.03},
    ("Sb", 3): {"O": 1.973, "F": 1.9, "Cl": 2.35},
    ("Sb", 5): {"O": 1.942, "F": 1.8, "Cl": 2.3},
    ("Sc", 3): {"O": 1.849, "F": 1.76, "Cl": 2.23},
    ("Se", 4): {"O": 1.811, "F": 1.73, "Cl": 2.22},
    ("Se", 6): {"O": 1.788, "F": 1.69, "Cl": 2.16},
    ("Si", 4): {"O": 1.624, "F": 1.58, "Cl": 2.03},
    ("Sm", 3): {"O": 2.088, "F": 1.977, "Cl": 2.466},
    ("Sn", 2): {"O": 1.984, "F": 1.925, "Cl": 2.36},
    ("Sn", 4): {"O": 1.905, "F": 1.84, "Cl": 2.28},
    ("Sr", 2): {"O": 2.118, "F": 2.019, "Cl": 2.51},
    ("Ta", 5): {"O": 1.92, "F": 1.88, "Cl": 2.3},
    ("Tb", 3): {"O": 2.049, "F": 1.936, "Cl": 2.427},
    ("Te", 4): {"O": 1.977, "F": 1.87, "Cl": 2.37},
    ("Te", 6): {"O": 1.917, "F": 1.82, "Cl": 2.3},
    ("Th", 4): {"O": 2.167, "F": 2.07, "Cl": 2.55},
    ("Ti", 3): {"O": 1.791, "F": 1.723, "Cl": 2.17},
    ("Ti", 4): {"O": 1.815, "F": 1.76, "Cl": 2.19},
    ("Tl", 1): {"O": 2.172, "F": 2.15, "Cl": 2.56},
    ("Tl", 3): {"O": 2.003, "F": 1.88, "Cl": 2.32},
    ("Tm", 3): {"O": 2, "F": 1.842, "Cl": 2.38},
    ("U", 4): {"O": 2.112, "F": 2.034, "Cl": 2.48},
    ("U", 6): {"O": 2.075, "F": 1.966, "Cl": 2.46},
    ("V", 3): {"O": 1.743, "F": 1.702, "Cl": 2.19},
    ("V", 4): {"O": 1.784, "F": 1.7, "Cl": 2.16},
    ("V", 5): {"O": 1.803, "F": 1.71, "Cl": 2.16},
    ("W", 6): {"O": 1.921, "F": 1.83, "Cl": 2.27},
    ("Y", 3): {"O": 2.014, "F": 1.904, "Cl": 2.4},
    ("Yb", 3): {"O": 1.985, "F": 1.875, "Cl": 2.371},
    ("Zn", 2): {"O": 1.704, "F": 1.62, "Cl": 2.01},
    ("Zr", 4): {"O": 1.937, "F": 1.854, "Cl": 2.33},
}

# Printed in italics in the original. The table's caption defines those as
# interpolated from the linear relations of its Table 1 rather than determined
# directly from structures -- the paper's own distinction, kept because it is
# the sort of thing a reader should be able to see.
_TABLE2_INTERPOLATED: set[tuple[str, int, str]] = {
    ("Ac", 3, "Cl"),
    ("Ac", 3, "F"),
    ("Am", 3, "Cl"),
    ("Am", 3, "F"),
    ("As", 5, "Cl"),
    ("Be", 2, "Cl"),
    ("Bi", 5, "Cl"),
    ("Bk", 3, "Cl"),
    ("Bk", 3, "F"),
    ("Br", 7, "Cl"),
    ("Br", 7, "F"),
    ("Ce", 4, "Cl"),
    ("Cl", 7, "Cl"),
    ("Cl", 7, "F"),
    ("Cm", 3, "Cl"),
    ("Cm", 3, "F"),
    ("Co", 3, "F"),
    ("Cr", 2, "O"),
    ("Cu", 1, "F"),
    ("Hf", 4, "Cl"),
    ("Hg", 1, "Cl"),
    ("Hg", 1, "F"),
    ("I", 5, "Cl"),
    ("I", 5, "F"),
    ("I", 7, "Cl"),
    ("In", 3, "Cl"),
    ("Mn", 3, "Cl"),
    ("Mn", 4, "Cl"),
    ("Mn", 7, "Cl"),
    ("Mo", 6, "F"),
    ("N", 5, "Cl"),
    ("N", 5, "F"),
    ("Os", 4, "Cl"),
    ("Os", 4, "F"),
    ("Pb", 4, "F"),
    ("Pt", 2, "F"),
    ("Rh", 3, "Cl"),
    ("S", 4, "Cl"),
    ("Sc", 3, "Cl"),
    ("Se", 6, "Cl"),
    ("Sn", 2, "Cl"),
    ("Te", 6, "Cl"),
    ("Th", 4, "Cl"),
    ("Ti", 3, "Cl"),
    ("Ti", 4, "Cl"),
    ("Tl", 1, "Cl"),
    ("V", 3, "Cl"),
    ("V", 4, "Cl"),
}

# Values FACET carried before this transcription that are NOT the values in
# Table 2, although Table 2 was the source cited for them. Five differ only in
# the decimal place the paper prints; five differ in substance, and Ag(I)-O is
# one the paper says explicitly that it changed from Brown & Altermatt (1985)
# by more than 0.02 A. FACET now uses Table 2, so that every value and its
# citation agree. They are recorded because anyone comparing a number with one
# computed earlier, or with another program, needs to know which was used --
# and because Brown & Altermatt is as legitimate a source as this one.
SUPERSEDED: dict[tuple[str, int, str], float] = {
    ("Ag", 1, "O"): 1.842,      # Table 2: 1.805, every valence x0.905
    ("B", 3, "F"): 1.281,       # Table 2: 1.31,  x1.082
    ("Ba", 2, "O"): 2.285,      # Table 2: 2.29,  x1.014
    ("Cs", 1, "O"): 2.417,      # Table 2: 2.42,  x1.008
    ("K", 1, "O"): 2.132,       # Table 2: 2.13,  x0.995
    ("Na", 1, "O"): 1.803,      # Table 2: 1.80,  x0.992
    ("P", 5, "O"): 1.617,       # Table 2: 1.604, x0.966
    ("Rb", 1, "O"): 2.263,      # Table 2: 2.26,  x0.992
    ("W", 6, "O"): 1.917,       # Table 2: 1.921, x1.011
    ("Y", 3, "O"): 2.019,       # Table 2: 2.014, x0.987
}


_FITTED: dict[tuple[str, int, str], float] = {
    # Bismuth to the heavier anions, from Table 3 of the same paper. Audited
    # against Sleight's compilation and reproduced to 0.001 A.
    ("Bi", 3, "Br"): 2.60, ("Bi", 3, "I"): 2.76,
    ("Bi", 3, "S"): 2.55, ("Bi", 3, "Se"): 2.67, ("Bi", 3, "Te"): 2.88,
    ("Bi", 3, "N"): 2.09,
}

# Anion formal charges assumed when pairing. The bond-valence model is defined
# per cation-anion pair and the anion charge is part of the pair's identity.
_ANION_OX = {"O": -2, "S": -2, "Se": -2, "Te": -2,
             "F": -1, "Cl": -1, "Br": -1, "I": -1, "N": -3, "H": -1}


# ---------------------------------------------------------------------------
# The O'Keeffe & Brese estimator
# ---------------------------------------------------------------------------
# O'Keeffe & Brese, J. Am. Chem. Soc. 113 (1991) 3226:
#
#     R0(ij) = ri + rj - ri*rj*(sqrt(ci) - sqrt(cj))**2 / (ci*ri + cj*rj)
#
# The per-element r and c parameters are read from pymatgen, which ships them as
# `pymatgen.analysis.bond_valence.BV_PARAMS`. pymatgen is a heavy import (about
# 1.7 s) so it is loaded lazily and cached -- FACET must not pay that cost at
# startup, and most sessions never touch a pair that needs the estimator.
_OKB_SOURCE = "O'Keeffe & Brese, J. Am. Chem. Soc. 113 (1991) 3226 (estimated)"
_okb_cache: dict[str, dict] | None = None


def _okb_params() -> dict[str, dict]:
    global _okb_cache
    if _okb_cache is None:
        try:
            from pymatgen.analysis.bond_valence import BV_PARAMS

            _okb_cache = {str(k): v for k, v in BV_PARAMS.items()}
        except Exception:
            _okb_cache = {}
    return _okb_cache


def estimate_r0(cation: str, anion: str) -> float | None:
    """Estimated R0 for any pair the O'Keeffe-Brese parameters cover."""
    p = _okb_params()
    a, b = p.get(elements.normalise(cation)), p.get(elements.normalise(anion))
    if not a or not b:
        return None
    ra, ca = a["r"], a["c"]
    rb, cb = b["r"], b["c"]
    denom = ca * ra + cb * rb
    if denom == 0:
        return None
    return ra + rb - ra * rb * (math.sqrt(ca) - math.sqrt(cb)) ** 2 / denom


# ---------------------------------------------------------------------------
# Parameter sets
# ---------------------------------------------------------------------------

class ParameterSet:
    """A named source of bond-valence parameters, with a documented fallback.

    Lookups are resolved in order: the fitted table, then the estimator. A set
    may be constructed with ``allow_estimated=False`` to make a missing fitted
    parameter an explicit failure rather than a silent downgrade -- useful when
    a result is going into a paper.
    """

    def __init__(self, name: str = "Brese & O'Keeffe 1991",
                 fitted: dict[tuple[str, int, str], float] | None = None,
                 source: str = _BO1991, b: float = B_DEFAULT,
                 allow_estimated: bool = True,
                 fitted_b: dict[tuple[str, int, str], float] | None = None,
                 anion_ox: dict[str, int] | None = None,
                 notes: list[str] | None = None):
        self.name = name
        self.source = source
        self.b = b
        self.allow_estimated = allow_estimated
        # Two layers for the built-in set: the hand-entered pairs, and the
        # whole of Table 2 behind them. A loaded file replaces both, because a
        # published compilation is meant to be used as a whole rather than
        # merged with another one.
        self._fitted = dict(_FITTED if fitted is None else fitted)
        self._table2 = dict(_TABLE2) if fitted is None else {}
        # Published sets give b per pair, not one universal value. Brese and
        # O'Keeffe's 0.37 A is a fitted average, and using it where a compilation
        # states something else changes every valence -- so a loaded set's own b
        # is kept per pair and only falls back to self.b where the file gave none.
        self._fitted_b = dict(fitted_b or {})
        self._anion_ox = dict(anion_ox or {})
        self.notes = list(notes or [])
        self._overrides: dict[tuple[str, int, str], BVParam] = {}

    # -- lookup ------------------------------------------------------------
    def get(self, cation: str, cation_ox: int | None, anion: str) -> BVParam | None:
        """The parameter for one pair, or None if nothing covers it."""
        c = elements.normalise(cation)
        a = elements.normalise(anion)
        ox = cation_ox if cation_ox is not None else elements.COMMON_OX.get(c)
        if ox is None:
            return None
        key = (c, int(ox), a)

        if key in self._overrides:
            return self._overrides[key]

        r0 = self._fitted.get(key)
        if r0 is not None:
            return BVParam(c, int(ox), a, self._anion_charge(a), r0,
                           self._fitted_b.get(key, self.b), self.source, True)

        row = self._table2.get((c, int(ox)))
        if row is not None and a in row:
            # The paper's own distinction: a value it printed in italics was
            # interpolated from its linear relations rather than determined
            # from structures. Still a published, fitted-set value -- but the
            # source says which it is, because it is the sort of thing a reader
            # should be able to see.
            interpolated = (c, int(ox), a) in _TABLE2_INTERPOLATED
            source = self.source + (" (interpolated in the original)"
                                    if interpolated else "")
            return BVParam(c, int(ox), a, self._anion_charge(a), row[a],
                           self.b, source, True)

        # The estimator has no oxidation-state dependence; it is a property of
        # the element pair. That is one of its limitations, and the reason a
        # fitted value is preferred wherever one exists.
        if not self.allow_estimated:
            return None
        est = estimate_r0(c, a)
        if est is None:
            return None
        return BVParam(c, int(ox), a, self._anion_charge(a), est, self.b,
                       _OKB_SOURCE, False)

    def _anion_charge(self, anion: str) -> int:
        return self._anion_ox.get(anion, _ANION_OX.get(anion, -1))

    def pairs(self):
        """Every fitted pair in the set, as ``(cation, ox, anion)`` keys."""
        out = set(self._fitted) | set(self._overrides)
        for (cation, ox), row in self._table2.items():
            out.update((cation, ox, anion) for anion in row)
        return sorted(out)

    def override(self, cation: str, cation_ox: int, anion: str,
                 r0: float, b: float | None = None, source: str = "user") -> None:
        """Pin a parameter by hand. Recorded as such, and reported as such."""
        c, a = elements.normalise(cation), elements.normalise(anion)
        self._overrides[(c, int(cation_ox), a)] = BVParam(
            c, int(cation_ox), a, self._anion_charge(a), float(r0),
            float(b if b is not None else self.b), source, True)

    def clear_overrides(self) -> None:
        self._overrides.clear()

    @property
    def n_fitted(self) -> int:
        """How many pairs the set covers without resorting to the estimator."""
        return len(self.pairs())

    def with_b(self, b: float) -> "ParameterSet":
        """The same set with one universal b, overriding any per-pair values.

        For asking what difference b makes -- which is a fair question, because
        the plateau width in log-valence is the distance gap divided by b, so b
        sets the scale on which two shells are resolved at all.
        """
        out = ParameterSet(self.name, self._fitted, self.source, b,
                           self.allow_estimated, fitted_b=None,
                           anion_ox=self._anion_ox, notes=self.notes)
        out._overrides = {k: replace(v, b=b) for k, v in self._overrides.items()}
        return out

    def __repr__(self) -> str:
        return (f"ParameterSet({self.name!r}, {self.n_fitted} fitted pairs, "
                f"b={self.b}, estimator={'on' if self.allow_estimated else 'off'})")


DEFAULT = ParameterSet()


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def valence(d, r0: float, b: float = B_DEFAULT):
    """v = exp((R0 - d) / b). Vectorised."""
    return np.exp((r0 - np.asarray(d, float)) / b)


def cutoff_for_valence(r0: float, v: float, b: float = B_DEFAULT) -> float:
    """The distance at which the bond valence falls to v.

    This is the function that replaces a distance cutoff. Feeding it the same v
    for every anion cuts every anion at the same bond strength.
    """
    if v <= 0:
        raise ValueError(
            f"a bond valence of {v:g} is reached only at infinite "
            "distance; the threshold must be greater than zero")
    return r0 - b * math.log(v)


def bvs(valences) -> float:
    """Bond-valence sum: the model's estimate of the cation's oxidation state."""
    v = np.asarray(valences, float)
    return float(v.sum()) if v.size else 0.0


def phi_index(vectors, valences) -> tuple[float, float, np.ndarray]:
    """Normalised bond-valence vector sum: ``|sum v_i u_i| / sum v_i``.

    Returns ``(|BVV| in v.u., phi, the BVV vector)``.

    phi is 0 when the bond valences balance in every direction and approaches 1
    when every bond points into one hemisphere, so it measures how one-sided an
    environment is -- for an ns2 cation, how stereochemically active the lone
    pair is.

    The reason FACET reports phi and not the bare vector sum: **phi is exactly
    invariant to an error in R0**. Shifting R0 by delta multiplies every v_i by
    exp(delta/b), a common factor that cancels between numerator and
    denominator. |BVV| has no such property. Since R0 carries about +-0.02 A
    even when fitted -- and several times that when estimated -- only the
    scale-free quantity can be compared across anions or across parameter sets.
    """
    v = np.asarray(valences, float)
    u = np.asarray(vectors, float)
    if v.size == 0:
        return 0.0, float("nan"), np.zeros(3)
    norms = np.linalg.norm(u, axis=1)
    norms[norms == 0] = 1.0
    u = u / norms[:, None]
    vec = (v[:, None] * u).sum(axis=0)
    mag = float(np.linalg.norm(vec))
    s = float(v.sum())
    return mag, (mag / s if s else float("nan")), vec


def valence_uncertainty(distances, r0: float, b: float,
                        r0_error: float = 0.02,
                        distance_errors=None) -> dict:
    """Propagate the known uncertainties into a bond-valence sum.

    Two sources, and they behave differently:

    * **R0** is shared by every bond of the pair, so its error is *systematic*
      -- it scales the whole sum by exp(delta/b) rather than averaging out.
      A fitted R0 carries about 0.02 A, which is 5.6 % at b = 0.37; an
      estimated one carries two to four times that.
    * **coordinate errors** are independent per bond, so they add in
      quadrature and partly cancel.

    Reporting them separately matters: the systematic part cannot be reduced by
    finding more bonds, and the random part can.
    """
    d = np.asarray(distances, float)
    if d.size == 0:
        return {"bvs": 0.0, "systematic": 0.0, "random": 0.0, "total": 0.0}

    v = np.exp((r0 - d) / b)
    total = float(v.sum())

    # a shift of R0 by delta multiplies every term by exp(delta/b)
    systematic = float(abs(total * (math.exp(r0_error / b) - 1.0)))

    if distance_errors is None:
        random = 0.0
    else:
        sigma = np.asarray(distance_errors, float)
        # dv/dd = -v/b, so each term contributes v*sigma/b
        random = float(np.sqrt(np.sum((v * sigma / b) ** 2)))

    return {
        "bvs": total,
        "systematic": systematic,
        "random": random,
        "total": float(math.hypot(systematic, random)),
        "r0_error": r0_error,
    }


def global_instability_index(discrepancies) -> float:
    """GII = sqrt(mean((BVS_i - V_i)^2)) over every site in the structure.

    A structure-wide measure of bond-valence strain. Values below about
    0.05 v.u. are unremarkable; above roughly 0.2 v.u. the structure is usually
    either strained, wrongly assigned, or wrong. FACET reports the number and
    the site that contributes most, and does not pass judgement on the strength
    of a fraction of the R0 uncertainty.
    """
    d = np.asarray(list(discrepancies), float)
    if d.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(d ** 2)))
