"""Coordination analysis for one site.

A coordination number is a function of a threshold, not a property of a
structure. This module computes that function -- the whole staircase, not one
value -- and then reports several independent definitions side by side so that
their agreement, or their disagreement, is itself visible.

The plateau identity
--------------------
Cut by bond valence and a useful thing falls out. Sorting a site's contacts by
valence, the coordination number is ``k`` for every threshold between the
valence of the k-th and the (k+1)-th contact. The width of that plateau in
log-valence is

    ln(v_k / v_(k+1)) = (d_(k+1) - d_k) / b

so **the plateau width is exactly the distance gap, divided by b**. A plateau
width and a gap in the distance distribution are the same measurement written
two ways.

FACET reports that width. It does not say what width is enough, because that
depends on the question being asked.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import bv, polyhedra
from .neighbors import Contacts
from .structure import Structure


@dataclass
class ContactRow:
    """One contact of the central site, with everything derived from it."""

    label: str
    element: str
    site_index: int
    atom_index: int
    distance: float
    vector: np.ndarray
    image: tuple[int, int, int]
    occupancy: float
    valence: float | None            # None when no parameter covers the pair
    param: bv.BVParam | None

    @property
    def has_valence(self) -> bool:
        return self.valence is not None

    @property
    def estimated_param(self) -> bool:
        return self.param is not None and not self.param.fitted


@dataclass
class Plateau:
    """A range of threshold over which the coordination number does not change."""

    cn: int
    v_high: float                    # threshold at the top of the plateau
    v_low: float                     # threshold at the bottom
    d_low: float                     # the distance of the innermost excluded contact
    d_high: float                    # the distance of the outermost included contact

    @property
    def width_decades(self) -> float:
        if self.v_low <= 0 or self.v_high <= 0:
            return float("inf")
        return math.log10(self.v_high / self.v_low)

    @property
    def width_angstrom(self) -> float:
        """The distance gap this plateau corresponds to."""
        return self.d_low - self.d_high

    def contains(self, v: float) -> bool:
        return self.v_low < v <= self.v_high


@dataclass
class SiteResult:
    """Everything FACET knows about one crystallographic site."""

    site_index: int
    label: str
    element: str
    ox: int | None
    ox_source: str
    wyckoff: str | None
    site_symmetry: str | None
    multiplicity: int | None

    contacts: list[ContactRow]
    v_bond: float
    v_list: float

    # --- the several coordination numbers ---------------------------------
    cn_valence: int = 0
    cn_ecoN: float = float("nan")
    d_ecoN: float = float("nan")
    cn_gap: int = 0
    gap_ratio: float = float("nan")
    cn_listed: int = 0
    # Contacts counted by occupancy rather than one each. For an ordered site
    # this equals cn_valence. For a disordered average structure it is the
    # number that means something: delta-Bi2O3 has 24 oxygen neighbours at one
    # distance, each a quarter occupied, which is six oxygens in any one cell.
    cn_occupancy: float = float("nan")

    # --- bond valence ------------------------------------------------------
    bvs: float = float("nan")
    bvs_listed: float = float("nan")
    bvv: float = float("nan")
    phi: float = float("nan")
    # phi recomputed over everything above the *listing* threshold. The gap
    # between the two is a direct measure of how cutoff-sensitive the
    # stereoactivity index is for this site; across the bismuth survey it
    # averaged 0.022, which is why phi can be quoted at all.
    phi_listed: float = float("nan")
    bvv_listed: float = float("nan")
    bvv_vector: np.ndarray = field(default_factory=lambda: np.zeros(3))
    valence_discrepancy: float | None = None
    f3: float = float("nan")
    # Uncertainty on the bond-valence sum, split by origin. The systematic
    # part comes from R0 and scales the whole sum; the random part comes from
    # the coordinates and partly cancels. See bv.valence_uncertainty.
    bvs_systematic: float = float("nan")
    bvs_random: float = float("nan")
    bvs_uncertainty: float = float("nan")

    # --- geometry ----------------------------------------------------------
    shape: dict = field(default_factory=dict)
    void_angle: float = float("nan")
    void_axis: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Shortest contact to another cation. Not part of the bond-valence sum,
    # but it is what explains a site whose sum falls far short: in a subvalent
    # or metallically bonded cluster the valence is in cation-cation bonds,
    # which a cation-anion model cannot see.
    nearest_cation: float | None = None
    nearest_cation_label: str | None = None

    # --- the staircase -----------------------------------------------------
    plateaus: list[Plateau] = field(default_factory=list)

    # --- provenance ---------------------------------------------------------
    uses_estimated_params: bool = False
    # Statements of fact about how the numbers were produced. Never advice, and
    # never a verdict on what they mean.
    notes: list[str] = field(default_factory=list)

    # -- derived ------------------------------------------------------------
    @property
    def bonds(self) -> list[ContactRow]:
        return [c for c in self.contacts
                if c.has_valence and c.valence > self.v_bond]

    @property
    def current_plateau(self) -> Plateau | None:
        for p in self.plateaus:
            if p.contains(self.v_bond):
                return p
        return None

    @property
    def plateau_decades(self) -> float:
        """Width, in decades of threshold, of the plateau the current threshold
        sits on. NaN when it sits exactly on a step edge.

        No judgement is attached to the value. What counts as a wide plateau
        depends on the question being asked, and is the reader's call.
        """
        p = self.current_plateau
        return p.width_decades if p else float("nan")

    @property
    def plateau_angstrom(self) -> float:
        """The distance gap the current plateau corresponds to, in angstrom."""
        p = self.current_plateau
        return p.width_angstrom if p else float("nan")

    def cn_at(self, v: float) -> int:
        return sum(1 for c in self.contacts
                   if c.has_valence and c.valence > v)

    def cn_occupancy_at(self, v: float) -> float:
        """Coordination number weighted by occupancy, at an arbitrary
        threshold."""
        return float(sum(c.occupancy for c in self.contacts
                         if c.has_valence and c.valence > v))

    @property
    def is_disordered(self) -> bool:
        """Whether any bonded neighbour is on a partially occupied site.

        Reported so that the difference between cn_valence and cn_occupancy is
        attributable rather than surprising.
        """
        return any(c.occupancy < 0.999 for c in self.bonds)

    def cn_within(self, d: float) -> int:
        return sum(1 for c in self.contacts if c.distance <= d)

    def summary(self) -> str:
        p = self.current_plateau
        plateau = (f"plateau {p.width_decades:.2f} dec / {p.width_angstrom:.3f} A"
                   if p else "on a step edge")
        return (f"{self.label} ({self.element}"
                f"{'' if self.ox is None else f'{self.ox:+d}'}): "
                f"CN {self.cn_valence}"
                f"{f' ({self.cn_occupancy:.2f} by occupancy)' if self.is_disordered else ''}"
                f" at {self.v_bond} v.u., "
                f"ECoN {self.cn_ecoN:.2f}, BVS {self.bvs:.2f}, "
                f"phi {self.phi:.3f}, {plateau}")


# ---------------------------------------------------------------------------

def analyse_site(structure: Structure, contacts: Contacts,
                 params: bv.ParameterSet | None = None,
                 v_bond: float = bv.V_BOND_DEFAULT,
                 v_list: float = bv.V_LIST_DEFAULT,
                 anions_only: bool = True) -> SiteResult:
    """Full analysis of one site from its cached contact list.

    `contacts` comes from :class:`facet.core.neighbors.NeighborFinder` and is
    never re-searched here -- moving `v_bond` and calling again is cheap, which
    is what the cutoff explorer relies on.
    """
    params = params or bv.DEFAULT
    site = structure.sites[contacts.center_site]

    # An anion centre is the same calculation with the roles exchanged: keep
    # the cations rather than the anions, and look the parameter up in the
    # other direction. Without both, an anion site comes back with no contacts
    # and a sum of nan, which is what it used to do.
    centre_is_anion = site.is_anion

    all_contacts = contacts
    if anions_only:
        contacts = (contacts.cations_only(structure) if centre_is_anion
                    else contacts.anions_only(structure))

    rows: list[ContactRow] = []
    used_estimate = False
    missing_pairs: set[str] = set()

    for i in range(len(contacts)):
        anion = contacts.elements[i]
        if centre_is_anion:
            # the pair is (the neighbouring cation, its charge, this anion)
            neighbour = structure.sites[int(contacts.neighbor_site[i])]
            p = params.get(neighbour.element, neighbour.ox, site.element)
            pair_name = f"{anion}-{site.element}"
        else:
            p = params.get(site.element, site.ox, anion)
            pair_name = f"{site.element}-{anion}"
        if p is None:
            missing_pairs.add(pair_name)
            valence = None
        else:
            valence = float(p.valence(contacts.distance[i]))
            used_estimate = used_estimate or not p.fitted
        rows.append(ContactRow(
            label=structure.atoms[contacts.neighbor_atom[i]].label,
            element=anion,
            site_index=int(contacts.neighbor_site[i]),
            atom_index=int(contacts.neighbor_atom[i]),
            distance=float(contacts.distance[i]),
            vector=contacts.vector[i],
            image=tuple(int(x) for x in contacts.image[i]),
            occupancy=float(contacts.occupancy[i]),
            valence=valence,
            param=p,
        ))

    result = SiteResult(
        site_index=contacts.center_site,
        label=site.label,
        element=site.element,
        ox=site.ox,
        ox_source=site.ox_source,
        wyckoff=site.wyckoff,
        site_symmetry=site.site_symmetry,
        multiplicity=site.multiplicity,
        contacts=[r for r in rows
                  if not r.has_valence or r.valence > v_list],
        v_bond=v_bond,
        v_list=v_list,
        uses_estimated_params=used_estimate,
    )

    # These are statements of fact about how the numbers were produced, not
    # advice about what to conclude from them.
    if missing_pairs:
        result.notes.append(
            "no bond-valence parameter for " + ", ".join(sorted(missing_pairs))
            + "; those contacts are listed without a valence")
    if used_estimate:
        result.notes.append(
            "some parameters are estimated from the O'Keeffe-Brese expression "
            "rather than fitted")
    if site.ox_source in ("common", "unset"):
        result.notes.append(
            f"oxidation state {site.ox} taken from the usual value for the "
            f"element, not from the file")

    _fill_valence_quantities(result)
    _fill_geometry(result)
    _fill_cation_contacts(structure, all_contacts, result)
    _note_empty_coordination(result)
    result.plateaus = plateaus(result)
    return result


def _fill_valence_quantities(r: SiteResult) -> None:
    bonded = [c for c in r.contacts if c.has_valence and c.valence > r.v_bond]
    listed = [c for c in r.contacts if c.has_valence and c.valence > r.v_list]

    r.cn_valence = len(bonded)
    r.cn_listed = len(listed)
    r.cn_occupancy = float(sum(c.occupancy for c in bonded))

    if listed:
        # The bond-valence sum is taken over everything above the *listing*
        # threshold, not the bond threshold. Valence does not stop existing
        # because a contact fell outside the coordination number, and truncating
        # the sum at the bond cutoff systematically under-counts it.
        r.bvs_listed = bv.bvs([c.valence * c.occupancy for c in listed])
        mag_l, phi_l, _ = bv.phi_index(
            np.array([c.vector for c in listed]),
            [c.valence * c.occupancy for c in listed])
        r.bvv_listed, r.phi_listed = mag_l, phi_l
    if bonded:
        r.bvs = bv.bvs([c.valence * c.occupancy for c in bonded])
        mag, phi, vec = bv.phi_index(
            np.array([c.vector for c in bonded]),
            [c.valence * c.occupancy for c in bonded])
        r.bvv, r.phi, r.bvv_vector = mag, phi, vec
        r.f3 = polyhedra.valence_fraction_in_shortest(
            [c.valence for c in bonded], 3)
        param = next((c.param for c in bonded if c.param), None)
        if param is not None:
            # an estimated R0 is worth roughly half the accuracy of a fitted
            # one, which is why the distinction is carried this far
            r0_error = 0.02 if param.fitted else 0.06
            unc = bv.valence_uncertainty(
                [c.distance for c in bonded], param.r0, param.b, r0_error)
            r.bvs_systematic = unc["systematic"]
            r.bvs_random = unc["random"]
            r.bvs_uncertainty = unc["total"]
        if r.ox is not None:
            # against the magnitude of the charge: a bond-valence sum is a
            # sum of positive terms, so an anion's -2 is matched by a sum near
            # +2. Subtracting the signed charge made every anion look over-
            # bonded by twice its charge, which was invisible while only
            # cations were analysed and feeds the global instability index and
            # the valence-discrepancy colouring as soon as they are not.
            r.valence_discrepancy = r.bvs - abs(r.ox)


def _fill_cation_contacts(structure: Structure, contacts: Contacts,
                          r: SiteResult) -> None:
    for i in range(len(contacts)):
        if structure.sites[contacts.neighbor_site[i]].is_anion:
            continue
        d = float(contacts.distance[i])
        if r.nearest_cation is None or d < r.nearest_cation:
            r.nearest_cation = d
            r.nearest_cation_label =                 structure.atoms[contacts.neighbor_atom[i]].label


def _fill_geometry(r: SiteResult) -> None:
    bonded = r.bonds
    if not bonded:
        return
    vectors = np.array([c.vector for c in bonded])
    distances = np.array([c.distance for c in bonded])

    r.shape = polyhedra.shape(vectors)
    r.cn_ecoN, r.d_ecoN = polyhedra.effective_cn(
        distances, [c.occupancy for c in bonded])
    r.void_angle, r.void_axis = polyhedra.void_cone(vectors)

    all_d = np.array([c.distance for c in r.contacts])
    r.cn_gap, r.gap_ratio = polyhedra.gap_split(all_d)


def plateaus(r: SiteResult) -> list[Plateau]:
    """Every coordination number this site can produce, and over what range.

    Walks the contacts outwards. Between the k-th and (k+1)-th contact the
    coordination number is exactly k for any threshold in that interval.
    """
    v = [(c.valence, c.distance) for c in r.contacts if c.has_valence]
    v.sort(key=lambda t: -t[0])
    if not v:
        return []

    out: list[Plateau] = []
    for k in range(1, len(v) + 1):
        v_high = v[k - 1][0]
        d_high = v[k - 1][1]
        if k < len(v):
            v_low, d_low = v[k][0], v[k][1]
        else:
            # The outermost plateau runs down to the listing threshold. Its
            # outer edge is the distance at which the valence reaches v_list,
            # not infinity -- the gap is finite and is what should be reported.
            v_low = r.v_list
            d_low = _distance_at(r, v_low, fallback=d_high)
        if v_low >= v_high:
            continue
        out.append(Plateau(cn=k, v_high=v_high, v_low=v_low,
                           d_low=d_low, d_high=d_high))
    return out


def _distance_at(r: SiteResult, v: float, fallback: float) -> float:
    """The distance at which a contact of this site would have valence v.

    Uses the parameter of the outermost contact that has one, which is the pair
    the plateau's outer edge is actually set by.
    """
    for c in reversed(r.contacts):
        if c.param is not None:
            return c.param.distance_for(v)
    return fallback


def _note_empty_coordination(r: SiteResult) -> None:
    """State the facts about a site with no bonds. Do not invent one.

    Some engines floor the coordination number at one. FACET reports zero,
    which is the answer to the question asked, and gives the numbers needed to
    see why.
    """
    if r.cn_valence or not r.contacts:
        return
    nearest = min((c for c in r.contacts if c.has_valence),
                  key=lambda c: c.distance, default=None)
    msg = f"no contact reaches the bond threshold of {r.v_bond} v.u."
    if nearest is not None:
        msg += (f"; nearest anion {nearest.label} at {nearest.distance:.3f} A "
                f"({nearest.valence:.3f} v.u.)")
    if r.nearest_cation is not None:
        msg += (f"; nearest cation {r.nearest_cation_label} at "
                f"{r.nearest_cation:.3f} A")
    r.notes.append(msg)


# ---------------------------------------------------------------------------

def staircase(r: SiteResult, n: int = 400,
              v_min: float | None = None,
              v_max: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """CN as a function of threshold, sampled for plotting.

    Returns ``(thresholds, cn)`` with the thresholds logarithmically spaced,
    because that is the axis on which plateau width means a distance gap.
    """
    v_min = v_min if v_min is not None else max(r.v_list * 0.5, 1e-4)
    grid = np.logspace(math.log10(v_min), math.log10(v_max), n)
    vals = np.array([c.valence for c in r.contacts if c.has_valence])
    if vals.size == 0:
        return grid, np.zeros(n, int)
    cn = (vals[None, :] > grid[:, None]).sum(axis=1)
    return grid, cn


def analyse_structure(structure: Structure, params: bv.ParameterSet | None = None,
                      v_bond: float = bv.V_BOND_DEFAULT,
                      v_list: float = bv.V_LIST_DEFAULT,
                      sites: list[int] | None = None,
                      cations_only: bool = True,
                      resolve_oxidation: bool = True) -> list[SiteResult]:
    """Analyse every site of a structure, sharing one neighbour search.

    Oxidation states are resolved first by default, because the bond-valence
    parameter depends on them: assuming the usual charge for the element turns
    a Bi(V) site into an apparently over-bonded Bi(III) one, and a mixed-valence
    structure into a uniform one. See :mod:`facet.core.oxidation`.
    """
    from .neighbors import NeighborFinder, search_radius_for

    params = params or bv.DEFAULT
    if resolve_oxidation:
        from . import oxidation

        oxidation.resolve(structure, params, v_bond, v_list)
    rmax = search_radius_for(structure, params, v_list)
    finder = NeighborFinder(structure, rmax=rmax)

    if sites is None:
        sites = structure.cation_sites if cations_only \
            else list(range(structure.n_sites))

    out = []
    for i in sites:
        try:
            contacts = finder.contacts_for_site(i)
        except ValueError:
            continue
        out.append(analyse_site(structure, contacts, params, v_bond, v_list))
    return out
