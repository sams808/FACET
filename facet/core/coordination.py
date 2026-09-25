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

so **the plateau width is exactly the distance gap, divided by b**. A wide
plateau and a real gap in the distance distribution are the same statement. A
coordination number sitting on a wide plateau is a property of the structure; a
coordination number sitting on a narrow step is a property of whoever chose the
cutoff.
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

    # --- honesty -----------------------------------------------------------
    uses_estimated_params: bool = False
    warnings: list[str] = field(default_factory=list)

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
    def cn_is_stable(self) -> bool:
        """Whether the reported CN sits on a plateau wide enough to be a
        property of the structure rather than of the threshold.

        Half a decade of threshold is 0.43 b, i.e. a distance gap of about
        0.16 A at b = 0.37. Below that, a different but defensible choice of
        threshold gives a different coordination number.
        """
        p = self.current_plateau
        return bool(p and p.width_decades >= 0.5)

    def cn_at(self, v: float) -> int:
        return sum(1 for c in self.contacts
                   if c.has_valence and c.valence > v)

    def cn_within(self, d: float) -> int:
        return sum(1 for c in self.contacts if c.distance <= d)

    def summary(self) -> str:
        p = self.current_plateau
        stable = ("stable over %.2f decades of threshold (a %.3f A gap)"
                  % (p.width_decades, p.width_angstrom)) if p else "no plateau"
        return (f"{self.label} ({self.element}"
                f"{'' if self.ox is None else f'{self.ox:+d}'}): "
                f"CN {self.cn_valence} at {self.v_bond} v.u., "
                f"ECoN {self.cn_ecoN:.2f}, BVS {self.bvs:.2f}, "
                f"phi {self.phi:.3f} -- {stable}")


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

    all_contacts = contacts
    if anions_only:
        contacts = contacts.anions_only(structure)

    rows: list[ContactRow] = []
    used_estimate = False
    missing_pairs: set[str] = set()

    for i in range(len(contacts)):
        anion = contacts.elements[i]
        p = params.get(site.element, site.ox, anion)
        if p is None:
            missing_pairs.add(f"{site.element}-{anion}")
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

    if missing_pairs:
        result.warnings.append(
            "no bond-valence parameter for " + ", ".join(sorted(missing_pairs))
            + "; those contacts are listed but carry no valence")
    if used_estimate:
        result.warnings.append(
            "some parameters are estimated from the O'Keeffe-Brese expression "
            "rather than fitted; treat the bond-valence sum accordingly")
    if site.ox_source in ("common", "unset"):
        result.warnings.append(
            f"oxidation state {site.ox} was assumed ({site.ox_source}), not "
            "read from the file")

    _fill_valence_quantities(result)
    _fill_geometry(result)
    _fill_cation_contacts(structure, all_contacts, result)
    _add_empty_coordination_warning(result)
    result.plateaus = plateaus(result)
    _add_stability_warning(result)
    return result


def _fill_valence_quantities(r: SiteResult) -> None:
    bonded = [c for c in r.contacts if c.has_valence and c.valence > r.v_bond]
    listed = [c for c in r.contacts if c.has_valence and c.valence > r.v_list]

    r.cn_valence = len(bonded)
    r.cn_listed = len(listed)

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
        if r.ox is not None:
            r.valence_discrepancy = r.bvs - r.ox


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


def _add_empty_coordination_warning(r: SiteResult) -> None:
    """Explain a site with no bonds rather than inventing one.

    Some engines floor the coordination number at one, on the grounds that a
    site with none is not a useful report. FACET does not: zero is the correct
    answer to the question asked, and it carries real information -- it says
    the site's valence is not in bonds to anions at all. The Bi9(5+) cluster in
    Bi12Cl14 is the standard example: every Bi-Cl contact is below 0.07 v.u.
    because the valence sits in Bi-Bi bonds near 3.07 A.
    """
    if r.cn_valence or not r.contacts:
        return
    nearest = min((c for c in r.contacts if c.has_valence),
                  key=lambda c: c.distance, default=None)
    msg = f"no contact reaches the bond threshold of {r.v_bond} v.u."
    if nearest is not None:
        msg += (f"; the nearest is {nearest.label} at {nearest.distance:.3f} A "
                f"({nearest.valence:.3f} v.u.)")
    if r.nearest_cation is not None:
        msg += (f". The nearest cation is {r.nearest_cation_label} at "
                f"{r.nearest_cation:.3f} A -- in a subvalent or metallically "
                "bonded site the valence is in cation-cation bonds, which a "
                "cation-anion bond-valence sum does not capture")
    r.warnings.append(msg)


def _add_stability_warning(r: SiteResult) -> None:
    p = r.current_plateau
    if p is None:
        if r.cn_valence:
            r.warnings.append(
                f"CN {r.cn_valence} sits exactly on a step edge; "
                "a negligible change of threshold changes it")
        return
    if p.width_decades < 0.5:
        alt = [q.cn for q in r.plateaus
               if q is not p and q.width_decades > p.width_decades]
        nearby = f" (CN {', '.join(str(a) for a in alt[:3])} sit on wider ones)" \
            if alt else ""
        r.warnings.append(
            f"CN {r.cn_valence} is stable over only {p.width_decades:.2f} "
            f"decades of threshold -- a gap of {p.width_angstrom:.3f} A -- so it "
            f"is a choice of cutoff more than a property of the site{nearby}")


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
