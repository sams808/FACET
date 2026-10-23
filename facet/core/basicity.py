"""Optical basicity: of a structure from its composition, and of each anion
from the bond valences it receives.

Duffy and Ingram's optical basicity (J. Non-Cryst. Solids 21 (1976) 373)
expresses the electron-donor power of the oxide ions of a material on a
scale anchored at CaO = 1. Each cation A lowers it through a *basicity
moderating parameter* gamma_A, related to the Pauling electronegativity by

    gamma = 1.36 (chi - 0.26)                                    (their eq.)

and a single oxide has Lambda = 1 / gamma. For a composition the bulk value
is their eq. (4),

    Lambda = 1 - sum_A (z_A r_A / 2) (1 - 1 / gamma_A),

z_A the oxidation number, r_A the number of A per oxygen: each cation lowers
the basicity in proportion to the share of the oxygen charge it neutralises.
Their section 1.4 applies the same equation to **one oxygen atom** -- the
microscopic optical basicity, written lambda -- with the share each
attached cation takes of that atom's charge, which they set from the
Pauling bond strength z / CN. A bridging Si-O-Si oxygen comes to 0.48, a
terminal Si-O to 0.74, Si-O-Al(IV) to 0.59.

FACET has that share directly: the bond valence s_ij of every contact of an
anion site. So, for an anion of formal charge -|z|,

    lambda = 1 - sum_j (s_j / |z|) (1 - 1 / gamma_j)

over its contacts, with s_j occupancy-weighted. Where the listed contacts
carry less than |z| the remainder counts as unmoderated (gamma = 1), as in
the equation above -- which is what makes a terminal oxygen more basic than
a bridging one. The sum over the contacts above the bond threshold is the
value reported beside the coordination number; the sum to the listing cut
is reported with it, as phi is.

Fluorides are on the same scale through Duffy's common oxide/fluoride
scale (J. Non-Cryst. Solids 109 (1989) 35): gamma for a fluoride
environment is 2.3 times the oxide value, so Lambda(NaF) = 0.50. Other
anions have no published scale here and are left out, with a note.

Four scales of gamma are offered. The default derives every cation's
gamma from Pauling's electronegativity as he tabulated it in 1960, which is
the table Duffy and Ingram's values reproduce (Si 2.09, Na 0.87, K 0.73,
Mg 1.28, B 2.36, P 2.50). Duffy's own later spectroscopic values (J. Chem.
Educ. 73 (1996) 1138, Table 1), Lebouteiller and Courtine's table for
cations by oxidation state and coordination (J. Solid State Chem. 137
(1998) 94, Table 2) and Dimitrov and Sakka's values from refractive index
and band gap (J. Appl. Phys. 79 (1996) 1736, Table I) are the other three;
a cation they do not list falls back to the electronegativity relation, and
every value says which it is. Duffy noted (1996) that the oxides of the
transition elements measure more basic than the electronegativity relation
gives; the measured scales are where that shows.

Nothing here judges a value. A number is reported with the scale, the
threshold and the shares that produced it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

# --- the relation -------------------------------------------------------------

GAMMA_SLOPE = 1.36
GAMMA_OFFSET = 0.26

# The anions optical basicity is defined for, with the factor on gamma that
# puts them on the oxide scale (Duffy 1989: fluoride gamma = 2.3 x oxide).
ANION_FACTORS: dict[str, float] = {"O": 1.0, "F": 2.3}
# The formal charge taken for an anion site that states none.
ANION_CHARGES: dict[str, int] = {"O": 2, "F": 1}

# Pauling's electronegativity scale as he tabulated it in 1960 (The Nature of
# the Chemical Bond, 3rd ed., Table 3-8), to one decimal. This is the table
# Duffy and Ingram's moderating parameters come from; the later revision
# (Allred 1961, two decimals, in facet.core.elements) gives Si 1.90 and
# Lambda(SiO2) = 0.45 where every optical-basicity table says 0.48. Pauling
# gave one range, 1.1-1.2, for lanthanum to lutetium; La-Yb are taken as 1.1
# and Lu as 1.2, and the entries say so.
PAULING_1960: dict[str, float] = {
    "H": 2.1,
    "Li": 1.0, "Be": 1.5, "B": 2.0, "C": 2.5, "N": 3.0, "O": 3.5, "F": 4.0,
    "Na": 0.9, "Mg": 1.2, "Al": 1.5, "Si": 1.8, "P": 2.1, "S": 2.5, "Cl": 3.0,
    "K": 0.8, "Ca": 1.0, "Sc": 1.3, "Ti": 1.5, "V": 1.6, "Cr": 1.6, "Mn": 1.5,
    "Fe": 1.8, "Co": 1.8, "Ni": 1.8, "Cu": 1.9, "Zn": 1.6, "Ga": 1.6,
    "Ge": 1.8, "As": 2.0, "Se": 2.4, "Br": 2.8,
    "Rb": 0.8, "Sr": 1.0, "Y": 1.2, "Zr": 1.4, "Nb": 1.6, "Mo": 1.8, "Tc": 1.9,
    "Ru": 2.2, "Rh": 2.2, "Pd": 2.2, "Ag": 1.9, "Cd": 1.7, "In": 1.7,
    "Sn": 1.8, "Sb": 1.9, "Te": 2.1, "I": 2.5,
    "Cs": 0.7, "Ba": 0.9,
    "La": 1.1, "Ce": 1.1, "Pr": 1.1, "Nd": 1.1, "Pm": 1.1, "Sm": 1.1,
    "Eu": 1.1, "Gd": 1.1, "Tb": 1.1, "Dy": 1.1, "Ho": 1.1, "Er": 1.1,
    "Tm": 1.1, "Yb": 1.1, "Lu": 1.2,
    "Hf": 1.3, "Ta": 1.5, "W": 1.7, "Re": 1.9, "Os": 2.2, "Ir": 2.2,
    "Pt": 2.2, "Au": 2.4, "Hg": 1.9, "Tl": 1.8, "Pb": 1.8, "Bi": 1.9,
    "Po": 2.0, "At": 2.2,
    "Fr": 0.7, "Ra": 0.9, "Ac": 1.1, "Th": 1.3, "Pa": 1.5, "U": 1.7,
    "Np": 1.3, "Pu": 1.3, "Am": 1.3, "Cm": 1.3, "Bk": 1.3, "Cf": 1.3,
    "Es": 1.3, "Fm": 1.3, "Md": 1.3, "No": 1.3,
}
_LANTHANIDE_RANGE = ("La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
                     "Dy", "Ho", "Er", "Tm", "Yb", "Lu")

PAULING_SOURCE = ("L. Pauling, The Nature of the Chemical Bond, 3rd ed. "
                  "(1960), Table 3-8; gamma = 1.36 (chi - 0.26) after J. A. "
                  "Duffy and M. D. Ingram, J. Non-Cryst. Solids 21 (1976) 373")
DUFFY_1996_SOURCE = ("J. A. Duffy, J. Chem. Educ. 73 (1996) 1138, Table 1 "
                     "(basicity moderating parameters from probe-ion spectra)")
DUFFY_1989_SOURCE = ("J. A. Duffy, J. Non-Cryst. Solids 109 (1989) 35, "
                     "Table 1 (fluoride gamma = 2.3 x oxide gamma)")
LEBOUTEILLER_SOURCE = ("A. Lebouteiller and P. Courtine, J. Solid State "
                       "Chem. 137 (1998) 94, Table 2 (known values of Lambda "
                       "by cation, oxidation state and coordination)")
DIMITROV_SOURCE = ("V. Dimitrov and S. Sakka, J. Appl. Phys. 79 (1996) 1736, "
                   "Table I (Lambda from the refractive index, or from the "
                   "band gap where no index is given)")


def gamma_from_electronegativity(chi: float) -> float:
    """Duffy and Ingram's basicity moderating parameter from a Pauling
    electronegativity: gamma = 1.36 (chi - 0.26)."""
    return GAMMA_SLOPE * (float(chi) - GAMMA_OFFSET)


# --- scales --------------------------------------------------------------------

@dataclass(frozen=True)
class Moderator:
    """One cation's basicity moderating parameter, with where it came from.

    ``measured`` is False when gamma was derived from the electronegativity
    relation rather than read from a table of measured or refractivity
    values. ``basicity`` is Lambda of the oxide, 1 / gamma.
    """

    element: str
    ox: int | None
    gamma: float
    source: str
    measured: bool
    coordination: int | None = None
    note: str = ""

    @property
    def basicity(self) -> float:
        return 1.0 / self.gamma

    @property
    def species(self) -> str:
        if self.ox is None:
            return self.element
        return f"{self.element}{abs(self.ox)}{'+' if self.ox >= 0 else '-'}"

    @property
    def origin(self) -> str:
        return "measured" if self.measured else "from electronegativity"


@dataclass(frozen=True)
class Scale:
    """A named set of moderating parameters.

    ``entries`` holds the table's own values by (element, oxidation state);
    a cation the table does not list is derived from
    :data:`PAULING_1960` through the relation, and the Moderator says so.
    """

    key: str
    name: str
    source: str
    entries: Mapping[tuple[str, int], Moderator] = field(default_factory=dict)

    def moderator(self, element: str, ox: int | None) -> Moderator | None:
        """The parameter for a cation, or None when nothing covers it."""
        if ox is not None:
            found = self.entries.get((element, int(ox)))
            if found is not None:
                return found
        chi = PAULING_1960.get(element)
        if chi is None:
            return None
        note = ("Pauling's table gives 1.1-1.2 for the lanthanides"
                if element in _LANTHANIDE_RANGE else "")
        return Moderator(element, None if ox is None else int(ox),
                         gamma_from_electronegativity(chi), PAULING_SOURCE,
                         False, note=note)

    @property
    def measured_species(self) -> list[str]:
        return [m.species for m in self.entries.values()]


def _measured(rows, source: str, *, gamma: bool) -> dict:
    out = {}
    for row in rows:
        element, ox, value, cn, note = row
        g = float(value) if gamma else 1.0 / float(value)
        out[(element, ox)] = Moderator(element, ox, g, source, True,
                                       coordination=cn, note=note)
    return out


# Duffy 1996, Table 1: gamma as printed. Phosphorus and sulphur are the
# 1996 values (3.0 and 4.0; the 1976 and 1989 tables had 2.50 for P).
_DUFFY_1996 = (
    ("Cs", 1, 0.60, None, ""), ("K", 1, 0.73, None, ""),
    ("Na", 1, 0.87, None, ""), ("Li", 1, 1.0, None, ""),
    ("Ba", 2, 0.87, None, ""), ("Sr", 2, 0.91, None, ""),
    ("Ca", 2, 1.00, None, "the anchor of the scale, CaO = 1"),
    ("Fe", 2, 1.0, None, ""), ("Mn", 2, 1.0, None, ""),
    ("Mg", 2, 1.3, None, ""), ("Al", 3, 1.65, None, ""),
    ("Si", 4, 2.1, None, ""), ("B", 3, 2.36, None, ""),
    ("H", 1, 2.5, None, ""), ("P", 5, 3.0, None, "1976 and 1989: 2.50"),
    ("S", 6, 4.0, None, ""),
)

# Lebouteiller and Courtine 1998, Table 2 ("known values"): Lambda as
# printed, with the coordination number of the cation and the paper's own
# flags (a value marked "hypothesis on Lambda" in the table).
_LEBOUTEILLER_1998 = (
    ("Al", 3, 0.60, 6, ""), ("B", 3, 0.42, 6, ""), ("Ba", 2, 1.20, 6, ""),
    ("Ca", 2, 1.00, 6, ""), ("Cs", 1, 1.70, 8, ""), ("K", 1, 1.40, 8, ""),
    ("Mg", 2, 0.78, 6, ""), ("P", 5, 0.33, 4, ""), ("Si", 4, 0.48, 4, ""),
    ("Sr", 2, 1.10, 6, ""), ("Na", 1, 1.15, 4, ""),
    ("Bi", 3, 1.19, 6, ""), ("Pb", 2, 1.18, 6, ""), ("Sb", 3, 1.18, 6, ""),
    ("Tl", 1, 1.22, 6, "a hypothesis on Lambda in the original"),
    ("Mo", 6, 0.52, 6, "a hypothesis on Lambda in the original"),
    ("V", 5, 0.63, 6, "a hypothesis on Lambda in the original"),
    ("Ti", 4, 0.75, 6, ""),
    ("W", 6, 0.51, 6, "a hypothesis on Lambda in the original"),
    ("Co", 2, 0.98, 6, "d7"), ("Cr", 3, 0.70, 6, "d3"),
    ("Cu", 2, 1.10, 6, "d9"), ("Fe", 3, 0.77, 6, "low spin d5"),
    ("Fe", 2, 1.00, 6, "high spin d6"), ("Mn", 2, 0.96, 6, "high spin d5"),
    ("Ni", 2, 0.91, 6, "d8"),
    ("Cd", 2, 1.12, 6, ""), ("Sb", 5, 0.98, 6, ""), ("Sn", 4, 0.87, 6, ""),
    ("Zn", 2, 1.10, 6, ""),
)

# Dimitrov and Sakka 1996, Table I: Lambda(n0) from the refractive index,
# or Lambda(Eg) from the band gap where the table gives no index.
_DIMITROV_1996 = (
    ("Li", 1, 0.87, None, "from n0"), ("Cu", 2, 1.08, None, "from n0"),
    ("Mg", 2, 0.69, None, "from n0"), ("Ca", 2, 1.00, None, "from n0"),
    ("Sr", 2, 1.10, None, "from n0"), ("Ba", 2, 1.21, None, "from n0"),
    ("Zn", 2, 1.03, None, "from n0"), ("Cd", 2, 1.10, None, "from n0"),
    ("B", 3, 0.43, None, "from n0"), ("Al", 3, 0.45, None, "from n0"),
    ("Ga", 3, 0.71, None, "from n0"), ("In", 3, 1.07, None, "from Eg"),
    ("Sc", 3, 0.87, None, "from Eg"), ("Y", 3, 0.99, None, "from n0"),
    ("Si", 4, 0.48, None, "from n0, quartz"),
    ("Ge", 4, 0.70, None, "from n0"), ("Sn", 4, 0.79, None, "from n0"),
    ("Pb", 2, 1.19, None, "from n0"),
    ("Ti", 4, 0.96, None, "from n0, rutile; anatase 1.02"),
    ("Zr", 4, 0.86, None, "from n0"), ("Ce", 4, 1.01, None, "from Eg"),
    ("Sb", 3, 1.14, None, "from n0"), ("Bi", 3, 1.19, None, "from Eg"),
    ("V", 5, 1.04, None, "from Eg"), ("Nb", 5, 1.05, None, "from Eg"),
    ("Ta", 5, 0.94, None, "from Eg"), ("Te", 4, 0.99, None, "from n0"),
    ("Mo", 6, 1.07, None, "from n0"), ("W", 6, 1.05, None, "from n0"),
    ("Mn", 2, 0.94, None, "from n0"), ("Fe", 3, 1.04, None, "from n0"),
    ("Co", 2, 0.98, None, "from Eg"), ("Ni", 2, 0.91, None, "from n0"),
)

PAULING = Scale("pauling",
                "Pauling electronegativity (Duffy and Ingram's relation)",
                PAULING_SOURCE)
DUFFY = Scale("duffy", "Duffy 1996, spectroscopic gamma", DUFFY_1996_SOURCE,
              _measured(_DUFFY_1996, DUFFY_1996_SOURCE, gamma=True))
LEBOUTEILLER = Scale("lebouteiller",
                     "Lebouteiller and Courtine 1998, by cation",
                     LEBOUTEILLER_SOURCE,
                     _measured(_LEBOUTEILLER_1998, LEBOUTEILLER_SOURCE,
                               gamma=False))
DIMITROV = Scale("dimitrov", "Dimitrov and Sakka 1996, from refractivity",
                 DIMITROV_SOURCE,
                 _measured(_DIMITROV_1996, DIMITROV_SOURCE, gamma=False))

SCALES: dict[str, Scale] = {s.key: s for s in (PAULING, DUFFY, LEBOUTEILLER,
                                                DIMITROV)}
SCALE_KEYS = tuple(SCALES)
DEFAULT_SCALE = "pauling"


def scale_named(key) -> Scale:
    """The scale with this key (a Scale is returned unchanged)."""
    if isinstance(key, Scale):
        return key
    try:
        return SCALES[str(key)]
    except KeyError:
        raise ValueError(f"no optical-basicity scale named {key!r}; the "
                         f"scales are {', '.join(SCALE_KEYS)}") from None


def oxide_basicity(element: str, ox: int | None = None,
                   scale=DEFAULT_SCALE, anion: str = "O") -> float:
    """Lambda of the single oxide (or fluoride) of one cation on a scale:
    1 / (f gamma), f the anion factor. NaN when the scale has nothing."""
    m = scale_named(scale).moderator(element, ox)
    factor = ANION_FACTORS.get(anion)
    if m is None or factor is None:
        return float("nan")
    return 1.0 / (factor * m.gamma)


# --- one anion ----------------------------------------------------------------

@dataclass
class Share:
    """One cation's part in an anion's basicity: the valence it delivers,
    the share of the anion's charge that is, and the lowering it causes."""

    label: str
    element: str
    ox: int | None
    site_index: int
    valence: float                  # occupancy-weighted, in v.u.
    share: float                    # valence / |z| of the anion
    moderator: Moderator | None
    bonded: bool                    # above the bond threshold
    lowering: float                 # share x (1 - 1 / (f gamma)); NaN without gamma

    @property
    def gamma(self) -> float:
        return float("nan") if self.moderator is None else self.moderator.gamma


@dataclass
class MicroBasicity:
    """The microscopic optical basicity of one anion site.

    ``value`` sums the contacts above the bond threshold, ``value_listed``
    every listed contact; both NaN when a contact's cation has no gamma on
    the scale (``missing`` names them) or the anion is not one the scale
    covers (``defined`` False, with the reason in ``notes``).
    """

    label: str
    element: str
    ox: int | None
    charge: float                   # |z| the shares are taken against
    factor: float                   # the anion factor on gamma
    scale: str
    defined: bool
    value: float = float("nan")
    value_listed: float = float("nan")
    sum_bonded: float = 0.0         # sum of the bonded valences, v.u.
    sum_listed: float = 0.0
    shares: list[Share] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def unmoderated(self) -> float:
        """The share of the charge the bonded contacts do not carry, which
        the equation counts at gamma = 1."""
        return 1.0 - self.sum_bonded / self.charge if self.charge else float("nan")

    @property
    def derived_species(self) -> list[str]:
        seen: dict[str, None] = {}
        for s in self.shares:
            if s.moderator is not None and not s.moderator.measured:
                seen.setdefault(s.moderator.species, None)
        return list(seen)

    @property
    def measured_species(self) -> list[str]:
        seen: dict[str, None] = {}
        for s in self.shares:
            if s.moderator is not None and s.moderator.measured:
                seen.setdefault(s.moderator.species, None)
        return list(seen)


def _cation_ox(contact) -> int | None:
    param = getattr(contact, "param", None)
    if param is not None and getattr(param, "cation_ox", None) is not None:
        return int(param.cation_ox)
    return None


def micro_basicity(result, scale=DEFAULT_SCALE, *,
                   v_bond: float | None = None) -> MicroBasicity:
    """lambda of one anion site from its ``SiteResult``.

    ``result`` is an anion-centred result (its contacts are the cations that
    reach it, with their valences). The bond threshold is the result's
    unless ``v_bond`` is given.
    """
    scale = scale_named(scale)
    v = float(result.v_bond if v_bond is None else v_bond)
    element = result.element
    factor = ANION_FACTORS.get(element)
    ox = result.ox
    charge = (abs(float(ox)) if ox is not None
              else float(ANION_CHARGES.get(element, 0)))
    out = MicroBasicity(result.label, element, ox, charge,
                        factor if factor is not None else float("nan"),
                        scale.key, defined=factor is not None and charge > 0)
    if factor is None:
        out.notes.append(
            f"optical basicity is defined for oxide and (through Duffy's "
            f"1989 common scale) fluoride anions; none is published here "
            f"for {element}")
        return out
    if charge <= 0:
        out.notes.append(f"{result.label} has no formal charge to take the "
                         "shares against")
        return out
    if ox is None:
        out.notes.append(f"formal charge -{int(charge)} taken for {element}, "
                         "the site states none")
    if ox is not None and ox > 0:
        out.notes.append(f"{result.label} is not an anion on its stated "
                         f"oxidation state {ox:+d}")
        out.defined = False
        return out

    skipped = 0
    for c in result.contacts:
        if not c.has_valence:
            skipped += 1
            continue
        cation_ox = _cation_ox(c)
        m = scale.moderator(c.element, cation_ox)
        valence = float(c.valence) * float(c.occupancy)
        share = valence / charge
        lowering = (float("nan") if m is None
                    else share * (1.0 - 1.0 / (factor * m.gamma)))
        out.shares.append(Share(c.label, c.element, cation_ox, int(c.site_index),
                                valence, share, m, bool(c.valence > v),
                                lowering))
        if m is None:
            species = (c.element if cation_ox is None
                       else f"{c.element}{abs(cation_ox)}+")
            if species not in out.missing:
                out.missing.append(species)
    if skipped:
        out.notes.append(f"{skipped} contact(s) without a valence are left "
                         "out of the shares")
    out.sum_bonded = sum(s.valence for s in out.shares if s.bonded)
    out.sum_listed = sum(s.valence for s in out.shares)
    if out.missing:
        out.notes.append("no basicity moderating parameter on the "
                         f"{scale.key} scale for " + ", ".join(out.missing)
                         + "; the electronegativity relation has none "
                         "either")
        return out
    out.value = 1.0 - sum(s.lowering for s in out.shares if s.bonded)
    out.value_listed = 1.0 - sum(s.lowering for s in out.shares)
    derived = out.derived_species
    if derived and scale.key != PAULING.key:
        out.notes.append("gamma from the electronegativity relation, not "
                         f"from the {scale.key} table, for "
                         + ", ".join(derived))
    return out


# --- the environment of a cation --------------------------------------------------

@dataclass
class EnvironmentBasicity:
    """The basicity of the anions a cation site is bonded to.

    ``value`` is the bond-valence-weighted mean of their lambda (each anion
    at the same bond threshold); ``value_host`` the same with the bonds
    from this site left out of each anion's sum, so the anions are seen as
    the rest of the structure leaves them. ``n_anions`` counts the bonded
    contacts whose anion has a lambda; ``n_undefined`` those whose anion has
    none (another anion, or a cation the scale lacks).
    """

    label: str
    element: str
    ox: int | None
    scale: str
    value: float = float("nan")
    value_host: float = float("nan")
    n_anions: int = 0
    n_undefined: int = 0
    anions: list[tuple[str, float, float, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def environment_basicity(result, anion_basicity: Mapping[int, MicroBasicity],
                         scale=DEFAULT_SCALE, *,
                         v_bond: float | None = None) -> EnvironmentBasicity:
    """The anions about a cation site, from the lambda of each
    (``anion_basicity`` by site index, at the same bond threshold)."""
    scale = scale_named(scale)
    v = float(result.v_bond if v_bond is None else v_bond)
    out = EnvironmentBasicity(result.label, result.element, result.ox,
                              scale.key)
    weights, values, hosts = [], [], []
    for c in result.contacts:
        if not c.has_valence or not c.valence > v:
            continue
        micro = anion_basicity.get(int(c.site_index))
        if micro is None or not micro.defined or math.isnan(micro.value):
            out.n_undefined += 1
            continue
        w = float(c.valence) * float(c.occupancy)
        own = sum(s.lowering for s in micro.shares
                  if s.bonded and s.site_index == result.site_index
                  and not math.isnan(s.lowering))
        host = micro.value + own
        weights.append(w)
        values.append(micro.value)
        hosts.append(host)
        out.anions.append((c.label, w, micro.value, host))
    out.n_anions = len(weights)
    if weights:
        total = sum(weights)
        out.value = sum(w * x for w, x in zip(weights, values)) / total
        out.value_host = sum(w * x for w, x in zip(weights, hosts)) / total
    if out.n_undefined:
        out.notes.append(f"{out.n_undefined} bonded contact(s) to an anion "
                         "with no lambda are left out")
    return out


# --- a composition --------------------------------------------------------------

@dataclass
class BulkRow:
    element: str
    ox: int
    count: float
    charge: float                   # z x count
    fraction: float                 # equivalent fraction X_A
    moderator: Moderator | None


@dataclass
class BulkBasicity:
    """Lambda of a composition, Duffy and Ingram's equation over the oxide
    (and fluoride) content.

    Each cation's equivalent fraction X_A = z_A n_A / sum(z n) over the
    cations; over an anion of factor f the value is
    1 - sum_A X_A (1 - 1 / (f gamma_A)), and with several covered anions the
    values are averaged with the anions' charge as weight -- the composition
    alone does not say which cations bond to which anion. Anions outside
    the scale are ``excluded``; a net charge is stated in the notes.
    """

    scale: str
    value: float = float("nan")
    rows: list[BulkRow] = field(default_factory=list)
    anions: list[tuple[str, float, float, float]] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def cation_charge(self) -> float:
        return sum(r.charge for r in self.rows)

    @property
    def anion_charge(self) -> float:
        return sum(a[2] for a in self.anions)


def bulk_basicity(composition: Mapping, scale=DEFAULT_SCALE) -> BulkBasicity:
    """Lambda of a composition given as counts by (element, oxidation
    state) -- ``{('Si', 4): 2, ('Na', 1): 2, ('O', -2): 5}``."""
    scale = scale_named(scale)
    out = BulkBasicity(scale.key)
    cations, anions = [], []
    for key, count in composition.items():
        element, ox = key
        n = float(count)
        if n <= 0:
            continue
        if ox is None:
            out.notes.append(f"{element} has no oxidation state and is left "
                             "out")
            continue
        (cations if ox > 0 else anions).append((element, int(ox), n))
    z_cat = sum(ox * n for _, ox, n in cations)
    for element, ox, n in cations:
        m = scale.moderator(element, ox)
        out.rows.append(BulkRow(element, ox, n, ox * n,
                                (ox * n / z_cat) if z_cat else float("nan"), m))
        if m is None:
            species = f"{element}{ox}+"
            if species not in out.missing:
                out.missing.append(species)
    q = 0.0
    for element, ox, n in anions:
        factor = ANION_FACTORS.get(element)
        if factor is None:
            out.excluded.append(element)
            continue
        q += abs(ox) * n
        out.anions.append((element, n, abs(ox) * n, factor))
    z_an = sum(abs(ox) * n for _, ox, n in anions)
    if out.excluded:
        out.notes.append("no optical-basicity scale for "
                         + ", ".join(sorted(set(out.excluded)))
                         + "; the value is over the oxide and fluoride "
                         "content only")
    if cations and anions and abs(z_cat - z_an) > 1e-6:
        out.notes.append(f"the composition carries a net charge of "
                         f"{z_cat - z_an:+.3f} with these oxidation states; "
                         "the equivalent fractions are taken over the "
                         "cation charge")
    if not cations or q <= 0:
        out.notes.append("no cation, or no oxide or fluoride anion, to take "
                         "a basicity over")
        return out
    if out.missing:
        out.notes.append("no basicity moderating parameter for "
                         + ", ".join(out.missing))
        return out
    total = 0.0
    for element, n, charge, factor in out.anions:
        lowering = sum(r.fraction * (1.0 - 1.0 / (factor * r.moderator.gamma))
                       for r in out.rows)
        total += (charge / q) * (1.0 - lowering)
    out.value = total
    derived = sorted({r.moderator.species for r in out.rows
                      if not r.moderator.measured})
    if derived and scale.key != PAULING.key:
        out.notes.append("gamma from the electronegativity relation, not "
                         f"from the {scale.key} table, for "
                         + ", ".join(derived))
    return out


def composition_by_species(structure) -> dict[tuple[str, int | None], float]:
    """Counts by (element, oxidation state) over a Structure's cell,
    occupancy and multiplicity weighted, as :func:`bulk_basicity` takes."""
    out: dict[tuple[str, int | None], float] = {}
    for site in structure.sites:
        mult = site.multiplicity if site.multiplicity else 1
        key = (site.element, site.ox)
        out[key] = out.get(key, 0.0) + float(site.occupancy) * mult
    return out


# --- a whole structure ----------------------------------------------------------

@dataclass
class StructureBasicity:
    """Every anion site's lambda, the environment of every cation site, the
    anion-charge-weighted means and the composition value, on one scale at
    one bond threshold."""

    scale: str
    v_bond: float
    bulk: BulkBasicity
    anions: dict[int, MicroBasicity] = field(default_factory=dict)
    environments: dict[int, EnvironmentBasicity] = field(default_factory=dict)
    mean: float = float("nan")
    mean_listed: float = float("nan")
    by_element: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    n_anion_atoms: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def difference(self) -> float:
        """The bond-valence mean less the composition value."""
        return self.mean - self.bulk.value


def structure_basicity(structure, results: Sequence, scale=DEFAULT_SCALE,
                       *, v_bond: float | None = None) -> StructureBasicity:
    """lambda of every anion site, the environment of every cation site and
    the means, from an analysis that covers the anions
    (``coordination.analyse_structure(..., cations_only=False)``).

    The mean weights each anion site by its atoms in the cell times its
    charge -- the weight the composition value gives it -- so the two are
    comparable: they agree when every cation's bonds carry exactly its
    formal charge, and differ by the departure of the bond-valence sums
    from the charges otherwise.
    """
    scale = scale_named(scale)
    results = list(results)
    if v_bond is None:
        v_bond = float(results[0].v_bond) if results else 0.0
    bulk = bulk_basicity(composition_by_species(structure), scale)
    out = StructureBasicity(scale.key, float(v_bond), bulk)
    by_index = {r.site_index: r for r in results}
    for index, site in enumerate(structure.sites):
        if not site.is_anion:
            continue
        r = by_index.get(index)
        if r is None:
            out.notes.append(f"{site.label} was not analysed; its lambda is "
                             "not known")
            continue
        out.anions[index] = micro_basicity(r, scale, v_bond=v_bond)
    weights: dict[str, list[tuple[float, float, float]]] = {}
    for index, micro in out.anions.items():
        if not micro.defined or math.isnan(micro.value):
            continue
        site = structure.sites[index]
        mult = site.multiplicity if site.multiplicity else 1
        atoms = float(site.occupancy) * mult
        weights.setdefault(micro.element, []).append(
            (atoms * micro.charge, micro.value, micro.value_listed))
    all_rows = [row for rows in weights.values() for row in rows]
    if all_rows:
        total = sum(w for w, _, _ in all_rows)
        out.mean = sum(w * x for w, x, _ in all_rows) / total
        out.mean_listed = sum(w * x for w, _, x in all_rows) / total
        out.n_anion_atoms = sum(w for w, _, _ in all_rows)
        for element, rows in weights.items():
            t = sum(w for w, _, _ in rows)
            out.by_element[element] = (
                sum(w * x for w, x, _ in rows) / t,
                sum(w * x for w, _, x in rows) / t, t)
    for index, site in enumerate(structure.sites):
        if site.is_anion:
            continue
        r = by_index.get(index)
        if r is None:
            continue
        out.environments[index] = environment_basicity(
            r, out.anions, scale, v_bond=v_bond)
    undefined = [m.label for m in out.anions.values()
                 if not m.defined or math.isnan(m.value)]
    if undefined:
        out.notes.append("no lambda for " + ", ".join(undefined)
                         + " (see the site's own notes)")
    return out


def site_basicity(result, results: Sequence, scale=DEFAULT_SCALE,
                  *, v_bond: float | None = None):
    """What the Site tab shows for one site: a :class:`MicroBasicity` for
    an anion, an :class:`EnvironmentBasicity` for a cation, from an
    analysis covering the anions (``results``) at the same threshold."""
    scale = scale_named(scale)
    v = float(result.v_bond if v_bond is None else v_bond)
    is_anion = (result.ox is not None and result.ox < 0) or (
        result.ox is None and result.element in ANION_FACTORS)
    if is_anion:
        return micro_basicity(result, scale, v_bond=v)
    by_index = {r.site_index: r for r in results}
    wanted = {int(c.site_index) for c in result.contacts
              if c.has_valence and c.valence > v}
    anions = {}
    for index in wanted:
        r = by_index.get(index)
        if r is None:
            continue
        if r.ox is not None and r.ox >= 0:
            continue
        anions[index] = micro_basicity(r, scale, v_bond=v)
    return environment_basicity(result, anions, scale, v_bond=v)


# --- an MD frame ------------------------------------------------------------------

@dataclass
class FrameBasicity:
    """lambda of every anion atom of a frame from its valence table.

    ``value`` and ``value_listed`` are (N,) arrays, NaN for a cation, for
    an anion outside the scale and for an anion with a contact whose cation
    has no gamma. ``classes`` (N,) counts each anion's bonded formers
    (-1 for a cation, or when no formers were named) as the glass
    speciation does: 0 free, 1 non-bridging, 2 bridging, 3 or more a
    tricluster.
    """

    scale: str
    v_bond: float
    value: np.ndarray
    value_listed: np.ndarray
    classes: np.ndarray
    bulk: BulkBasicity
    n_missing_gamma: int = 0
    missing: tuple[str, ...] = ()
    excluded: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def frame_basicity(table, scale=DEFAULT_SCALE, *, v_bond: float,
                   formers=None) -> FrameBasicity:
    """Every anion of a frame, from a ``bulk.ValenceTable`` (its listed
    counter-ion contacts, valence-descending, with the frame's elements and
    oxidation states)."""
    scale = scale_named(scale)
    v_bond = float(v_bond)
    elements = np.asarray(table.elements).astype(str)
    ox = np.asarray(table.ox, dtype=np.int64)
    is_anion = np.asarray(table.is_anion, dtype=bool)
    n = elements.shape[0]
    symbols = sorted(set(elements.tolist()))
    gamma_of: dict[str, float] = {}
    factor_of: dict[str, float] = {}
    missing: list[str] = []
    excluded: list[str] = []
    for symbol in symbols:
        rows = np.flatnonzero(elements == symbol)
        state = int(ox[rows[0]])
        if state >= 0:
            m = scale.moderator(symbol, state)
            if m is None:
                missing.append(f"{symbol}{state}+")
                gamma_of[symbol] = float("nan")
            else:
                gamma_of[symbol] = m.gamma
        else:
            f = ANION_FACTORS.get(symbol)
            if f is None:
                excluded.append(symbol)
            factor_of[symbol] = float("nan") if f is None else f
    gamma_atom = np.array([gamma_of.get(e, float("nan")) for e in elements])
    factor_atom = np.array([factor_of.get(e, float("nan")) for e in elements])
    charge = np.abs(ox).astype(np.float64)

    v = np.asarray(table.v_vu, dtype=np.float64)
    nbr = np.asarray(table.nbr, dtype=np.int64)
    width = v.shape[1] if v.ndim == 2 else 0
    n_listed = np.asarray(table.n_listed, dtype=np.int64)
    valid = (np.arange(width)[None, :] < n_listed[:, None]) if width else \
        np.zeros((n, 0), bool)
    safe = np.where(valid, nbr, 0)
    g = gamma_atom[safe] if width else np.zeros((n, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(valid, v / np.where(charge > 0, charge, np.nan)[:, None],
                         0.0)
        lowering = share * (1.0 - 1.0 / (factor_atom[:, None] * g))
    lowering = np.where(valid, lowering, 0.0)
    unknown = valid & ~np.isfinite(lowering)
    value_listed = 1.0 - lowering.sum(axis=1)
    bonded = valid & (v > v_bond)
    value = 1.0 - np.where(bonded, lowering, 0.0).sum(axis=1)
    undefined = ~is_anion | ~np.isfinite(factor_atom) | unknown.any(axis=1) | \
        ~(charge > 0)
    value[undefined] = np.nan
    value_listed[undefined] = np.nan
    n_missing = int((is_anion & np.isfinite(factor_atom)
                     & unknown.any(axis=1)).sum())

    classes = np.full(n, -1, dtype=np.int64)
    if formers:
        former = np.isin(elements, [str(f) for f in formers])
        counts = (bonded & former[safe]).sum(axis=1) if width else \
            np.zeros(n, np.int64)
        classes[is_anion] = counts[is_anion]

    composition: dict[tuple[str, int], float] = {}
    for symbol in symbols:
        rows = np.flatnonzero(elements == symbol)
        composition[(symbol, int(ox[rows[0]]))] = float(rows.size)
    bulk = bulk_basicity(composition, scale)
    notes = []
    if missing:
        notes.append("no basicity moderating parameter on the "
                     f"{scale.key} scale for " + ", ".join(missing)
                     + f"; {n_missing} anion(s) with such a contact have no "
                     "lambda")
    if excluded:
        notes.append("no optical-basicity scale for " + ", ".join(excluded))
    return FrameBasicity(scale.key, v_bond, value, value_listed, classes, bulk,
                         n_missing, tuple(missing), tuple(excluded),
                         tuple(notes))


CLASS_NAMES = {0: "free", 1: "NBO", 2: "BO", 3: "tricluster"}


def class_name(count: int) -> str:
    """The glass speciation name of an anion bonded to ``count`` formers."""
    return CLASS_NAMES[min(int(count), 3)]
