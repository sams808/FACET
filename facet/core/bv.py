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
        """The distance at which the bond valence falls to v."""
        return self.r0 - self.b * math.log(v)


# ---------------------------------------------------------------------------
# Fitted parameters
# ---------------------------------------------------------------------------
# Brese & O'Keeffe, Acta Cryst. B47 (1991) 192, Table 2. b = 0.37 throughout.
#
# This is deliberately NOT a copy of the whole published table. It holds the
# pairs FACET has been validated on -- the bismuth set was audited against
# Sleight's independent compilation (Physica C 514 (2015) 152) during the
# 2026-08-31 Na-Bi-O work and reproduced to 0.001 A -- plus common pairs whose
# values are widely reproduced. Anything absent falls through to the estimator,
# which is a better outcome than a mistyped "fitted" value.
#
# `tests/test_bv.py` cross-checks every entry here against the estimator and
# fails on any that disagree by more than 0.20 A, which catches transcription
# errors without requiring the estimator to be accurate.
_BO1991 = "Brese & O'Keeffe, Acta Cryst. B47 (1991) 192, Table 2"

_FITTED: dict[tuple[str, int, str], float] = {
    # --- bismuth: the audited set -------------------------------------------
    ("Bi", 3, "O"): 2.09, ("Bi", 5, "O"): 2.06,
    ("Bi", 3, "F"): 1.99, ("Bi", 5, "F"): 1.97,
    ("Bi", 3, "Cl"): 2.48, ("Bi", 3, "Br"): 2.60, ("Bi", 3, "I"): 2.76,
    ("Bi", 3, "S"): 2.55, ("Bi", 3, "Se"): 2.67, ("Bi", 3, "Te"): 2.88,
    ("Bi", 3, "N"): 2.09,
    # --- alkali / alkaline earth oxides -------------------------------------
    ("Li", 1, "O"): 1.466, ("Na", 1, "O"): 1.803, ("K", 1, "O"): 2.132,
    ("Rb", 1, "O"): 2.263, ("Cs", 1, "O"): 2.417,
    ("Mg", 2, "O"): 1.693, ("Ca", 2, "O"): 1.967, ("Sr", 2, "O"): 2.118,
    ("Ba", 2, "O"): 2.285,
    ("Na", 1, "F"): 1.677, ("Ca", 2, "F"): 1.842,
    # --- network formers ----------------------------------------------------
    ("B", 3, "O"): 1.371, ("Si", 4, "O"): 1.624, ("P", 5, "O"): 1.617,
    ("Ge", 4, "O"): 1.748, ("As", 5, "O"): 1.767, ("S", 6, "O"): 1.624,
    ("C", 4, "O"): 1.390, ("Al", 3, "O"): 1.651, ("Ga", 3, "O"): 1.730,
    ("Si", 4, "F"): 1.58, ("B", 3, "F"): 1.281,
    # --- transition metals --------------------------------------------------
    ("Ti", 4, "O"): 1.815, ("V", 5, "O"): 1.803, ("Cr", 3, "O"): 1.724,
    ("Mn", 2, "O"): 1.790, ("Fe", 3, "O"): 1.759, ("Fe", 2, "O"): 1.734,
    ("Co", 2, "O"): 1.692, ("Ni", 2, "O"): 1.654, ("Cu", 2, "O"): 1.679,
    ("Zn", 2, "O"): 1.704, ("Zr", 4, "O"): 1.937, ("Nb", 5, "O"): 1.911,
    ("Mo", 6, "O"): 1.907, ("Ta", 5, "O"): 1.920, ("W", 6, "O"): 1.917,
    ("Y", 3, "O"): 2.019, ("Sc", 3, "O"): 1.849, ("Hf", 4, "O"): 1.923,
    # --- heavy main group ---------------------------------------------------
    ("Pb", 2, "O"): 2.112, ("Sn", 4, "O"): 1.905, ("Sb", 5, "O"): 1.942,
    ("In", 3, "O"): 1.902, ("Cd", 2, "O"): 1.904, ("Ag", 1, "O"): 1.842,
    ("La", 3, "O"): 2.172, ("Th", 4, "O"): 2.167, ("U", 6, "O"): 2.075,
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
                 allow_estimated: bool = True):
        self.name = name
        self.source = source
        self.b = b
        self.allow_estimated = allow_estimated
        self._fitted = dict(_FITTED if fitted is None else fitted)
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
            return BVParam(c, int(ox), a, _ANION_OX.get(a, -1), r0, self.b,
                           self.source, True)

        # The estimator has no oxidation-state dependence; it is a property of
        # the element pair. That is one of its limitations, and the reason a
        # fitted value is preferred wherever one exists.
        if not self.allow_estimated:
            return None
        est = estimate_r0(c, a)
        if est is None:
            return None
        return BVParam(c, int(ox), a, _ANION_OX.get(a, -1), est, self.b,
                       _OKB_SOURCE, False)

    def override(self, cation: str, cation_ox: int, anion: str,
                 r0: float, b: float | None = None, source: str = "user") -> None:
        """Pin a parameter by hand. Recorded as such, and reported as such."""
        c, a = elements.normalise(cation), elements.normalise(anion)
        self._overrides[(c, int(cation_ox), a)] = BVParam(
            c, int(cation_ox), a, _ANION_OX.get(a, -1), float(r0),
            float(b if b is not None else self.b), source, True)

    def clear_overrides(self) -> None:
        self._overrides.clear()

    @property
    def n_fitted(self) -> int:
        return len(self._fitted)

    def with_b(self, b: float) -> "ParameterSet":
        out = ParameterSet(self.name, self._fitted, self.source, b,
                           self.allow_estimated)
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
