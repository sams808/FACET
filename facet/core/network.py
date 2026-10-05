"""A priori bond valences, and where bond-length variation comes from.

An observed bond valence says what a bond *is*. An **a priori** bond valence
says what it would be if nothing but the bond topology decided it -- the ideal
valences intrinsic to a structure, fixed by the formal valences of its sites and
by which site is bonded to which, and by nothing else. The difference between
the two is the part of a structure that its connectivity does not explain.

Gagné & Hawthorne (*IUCrJ* **7** (2020) 581) use that difference to separate the
causes of bond-length variation, with two indices:

* **Δ_topol**, the spread of the a priori valences about their mean -- variation
  the topology requires, whatever else is going on;
* **Δ_cryst**, how far the observed valences sit from the a priori ones --
  variation the topology does not account for, which is where electronic
  effects such as lone-pair stereoactivity live, along with the ordinary
  consequences of packing a structure into a periodic lattice.

Both are mean absolute deviations in valence units. Over the transition-metal
oxides of that paper, ⟨Δ_topol⟩ = 0.102 and ⟨Δ_cryst⟩ = 0.113 v.u.

HOW THE A PRIORI VALENCES ARE FOUND
-----------------------------------
The bond-valence model gives two rules. The *valence-sum rule*: the valences of
the bonds at a site sum to the magnitude of its formal valence. The *loop rule*:
around any closed path in the bond network, the valences taken alternately
positive and negative sum to zero. Written out for a structure, those are the
network equations, and solving them gives the a priori valences.

Written out is the awkward part: the loop equations need a cycle basis, and
choosing one by hand is what makes the published calculations laborious. This
module does not write them out. Rutherford (*Acta Cryst.* B46 (1990) 289)
observed that the two rules are Kirchhoff's two laws for a resistor network in
which every bond is a unit resistance, every cation injects a current equal to
its formal valence and every anion draws one off. The bond valences are then the
currents, and they follow from one weighted Laplacian solve with no cycle basis
to choose.

Checked against the worked example Gagné & Hawthorne set out in full (§4.1.2, a
*Pnma* perovskite): this reaches their answer to 2e-16.

WHAT IT NEEDS, AND WHEN IT CANNOT ANSWER
----------------------------------------
A bond topology that closes: the number of A-X bonds in a cell counted from the
A end has to equal the number counted from the X end, and the cell has to be
charge balanced. Real files often fail one or the other -- a partially occupied
site, a missing hydrogen, an oxidation state that does not balance. Those
structures get :class:`Topology` with ``closes`` false and a reason, rather than
an answer that looks like the others.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .structure import Structure

# How far the two counts of one bond type may differ before the topology is
# called open. Not zero, because occupancies make the counts fractional.
COUNT_TOLERANCE = 1e-6
# How far from neutral a cell may be. 0.01 e per cell is below what rounding a
# published occupancy to two decimals can produce.
CHARGE_TOLERANCE = 0.01


@dataclass
class Topology:
    """Which site is bonded to which, and how many times per unit cell.

    ``bonds`` is keyed by ``(cation site index, anion site index)`` and holds
    the number of bonds of that kind in one cell, counted with occupancy. The
    key is ordered cation first, because the bond-valence model is defined on
    that pair.
    """

    bonds: dict[tuple[int, int], float] = field(default_factory=dict)
    charge: dict[int, float] = field(default_factory=dict)
    multiplicity: dict[int, float] = field(default_factory=dict)
    labels: dict[int, str] = field(default_factory=dict)
    closes: bool = True
    reasons: list[str] = field(default_factory=list)

    @property
    def sites(self) -> list[int]:
        out: set[int] = set()
        for cation, anion in self.bonds:
            out.add(cation)
            out.add(anion)
        return sorted(out)

    def net_charge(self) -> float:
        return sum(self.charge.get(s, 0.0) for s in self.sites)

    def describe(self) -> str:
        rows = [f"{self.labels.get(c, c)}-{self.labels.get(a, a)}: {n:g}"
                for (c, a), n in sorted(self.bonds.items())]
        return "; ".join(rows)


def topology_of(structure: Structure, results, v_bond: float) -> Topology:
    """The bond topology of a structure, from an analysis that includes anions.

    ``results`` must cover the anions as well as the cations -- that is what
    gives the count from the other end, and without it there is nothing to check
    the cation counts against. :func:`facet.core.coordination.analyse_structure`
    with ``cations_only=False`` produces it.
    """
    topology = Topology()
    for site_index, site in enumerate(structure.sites):
        topology.multiplicity[site_index] = float(site.multiplicity or 1)
        topology.labels[site_index] = site.label
        charge = 0.0 if site.ox is None else float(site.ox)
        topology.charge[site_index] = (charge * float(site.occupancy)
                                       * float(site.multiplicity or 1))

    # counted from each end, so that the two can be compared
    counted: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
    saw_anion_centre = False
    for r in results:
        centre = r.site_index
        site = structure.sites[centre]
        if site.is_anion:
            saw_anion_centre = True
        mult = float(site.multiplicity or 1)
        per_neighbour: dict[int, float] = defaultdict(float)
        for contact in r.bonds:
            if contact.valence is None or contact.valence < v_bond:
                continue
            # occupancy of the far end: half a neighbour is half a bond
            per_neighbour[contact.site_index] += float(contact.occupancy)
        for other, n in per_neighbour.items():
            pair = ((centre, other) if not site.is_anion else (other, centre))
            counted[pair][centre] = mult * n * float(site.occupancy)

    if not saw_anion_centre:
        topology.closes = False
        topology.reasons.append(
            "the analysis covered only the cations, so there is nothing to "
            "check the bond counts against")

    for pair, ends in sorted(counted.items()):
        values = list(ends.values())
        topology.bonds[pair] = float(np.mean(values))
        if len(values) == 2 and abs(values[0] - values[1]) > COUNT_TOLERANCE:
            topology.closes = False
            cation, anion = pair
            topology.reasons.append(
                f"{topology.labels[cation]}-{topology.labels[anion]}: "
                f"{values[0]:g} bonds per cell counted from one end and "
                f"{values[1]:g} from the other")

    net = topology.net_charge()
    if abs(net) > CHARGE_TOLERANCE:
        topology.closes = False
        topology.reasons.append(
            f"the cell carries a net charge of {net:+.3f} e, so the valences "
            f"cannot sum to the formal charges at every site at once")
    return topology


def a_priori_valences(topology: Topology) -> dict[tuple[int, int], float]:
    """Solve the network equations: one valence per bond type.

    The resistor equivalence, as described in the module docstring. Each bond
    type is a conductance equal to the number of such bonds in the cell, each
    site a current source of its total formal charge, and the valence of one
    bond of a type is the potential difference across it.

    Solved per connected component, because a structure whose bond network
    falls into pieces -- a hydrate with isolated water, say -- has a separate
    set of equations for each, and each has to be charge balanced on its own.
    """
    sites = topology.sites
    if not sites:
        return {}
    index = {s: i for i, s in enumerate(sites)}
    n = len(sites)

    laplacian = np.zeros((n, n))
    for (cation, anion), count in topology.bonds.items():
        i, j = index[cation], index[anion]
        laplacian[i, i] += count
        laplacian[j, j] += count
        laplacian[i, j] -= count
        laplacian[j, i] -= count
    current = np.array([topology.charge.get(s, 0.0) for s in sites], float)

    # one grounded node per component: only differences matter, and a component
    # that is not charge balanced has no solution rather than a wrong one
    for component in _components(topology, sites, index):
        rows = sorted(component)
        if abs(current[rows].sum()) > CHARGE_TOLERANCE:
            raise ValueError(
                "a part of the bond network carries a net charge of "
                f"{current[rows].sum():+.3f} e, so its network equations have "
                "no solution")
        ground = rows[0]
        laplacian[ground, :] = 0.0
        laplacian[ground, ground] = 1.0
        current[ground] = 0.0

    potential = np.linalg.solve(laplacian, current)
    return {(cation, anion): float(potential[index[cation]]
                                   - potential[index[anion]])
            for (cation, anion) in topology.bonds}


def _components(topology: Topology, sites, index) -> list[set[int]]:
    """The connected pieces of the bond network, as row indices."""
    neighbours: dict[int, set[int]] = {index[s]: set() for s in sites}
    for cation, anion in topology.bonds:
        neighbours[index[cation]].add(index[anion])
        neighbours[index[anion]].add(index[cation])
    unseen = set(neighbours)
    out = []
    while unseen:
        start = unseen.pop()
        group, stack = {start}, [start]
        while stack:
            node = stack.pop()
            for other in neighbours[node]:
                if other not in group:
                    group.add(other)
                    unseen.discard(other)
                    stack.append(other)
        out.append(group)
    return out


@dataclass
class SiteIndices:
    """What the two indices say about one coordination polyhedron."""

    site_index: int
    label: str
    element: str
    ox: int | None
    coordination: int
    a_priori: list[float]
    observed: list[float]
    delta_topol: float
    delta_cryst: float

    @property
    def mean_a_priori(self) -> float:
        return float(np.mean(self.a_priori)) if self.a_priori else float("nan")


def indices(a_priori, observed) -> tuple[float, float]:
    """``(Δ_topol, Δ_cryst)`` for one polyhedron, in valence units.

    Both are mean absolute deviations: Δ_topol of the a priori valences about
    their own mean, Δ_cryst of the observed valences about the a priori ones.
    The two lists are bond for bond and must be in the same order.

    Checked against the values Gagné & Hawthorne quote for individual
    polyhedra; see tests/test_network.py.
    """
    a = np.asarray(a_priori, float)
    o = np.asarray(observed, float)
    if a.size == 0 or a.size != o.size:
        return float("nan"), float("nan")
    return (float(np.abs(a - a.mean()).mean()),
            float(np.abs(o - a).mean()))


def analyse(structure: Structure, results, v_bond: float) -> list[SiteIndices]:
    """Δ_topol and Δ_cryst for every cation polyhedron of a structure.

    ``results`` must include the anions, because the topology is checked from
    both ends. Raises if the topology does not close -- a number produced from
    a bond network that does not balance would look like the others and mean
    nothing.
    """
    topology = topology_of(structure, results, v_bond)
    if not topology.closes:
        raise ValueError("; ".join(topology.reasons))
    valences = a_priori_valences(topology)

    out = []
    for r in results:
        site = structure.sites[r.site_index]
        if site.is_anion:
            continue
        a_priori, observed = [], []
        for contact in r.bonds:
            if contact.valence is None or contact.valence < v_bond:
                continue
            key = (r.site_index, contact.site_index)
            if key not in valences:
                continue
            a_priori.append(valences[key])
            observed.append(float(contact.valence) * float(contact.occupancy))
        if not a_priori:
            continue
        topol, cryst = indices(a_priori, observed)
        out.append(SiteIndices(
            site_index=r.site_index, label=r.label, element=r.element,
            ox=r.ox, coordination=len(a_priori),
            a_priori=a_priori, observed=observed,
            delta_topol=topol, delta_cryst=cryst))
    return out
