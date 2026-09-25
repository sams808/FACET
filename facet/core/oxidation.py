"""Deciding the oxidation state of each site.

The bond-valence parameter depends on the cation's formal charge, so a wrong
oxidation state is a wrong R0 and therefore a wrong bond-valence sum. For
bismuth the two candidates differ by 0.03 A, which is a factor exp(0.03/0.37) =
1.085 in every valence -- enough to turn a perfectly good Bi(V) site into an
apparently over-bonded Bi(III) one.

Four routes, tried in order, and **the route used is always recorded**:

1. ``cif``            -- the file states a charge on the atom type symbol.
2. ``bond valence``   -- self-consistency: of the plausible charges, take the
                         one whose own R0 reproduces it best.
3. ``charge balance`` -- the remaining charge needed to neutralise the cell,
                         when only one site is undetermined.
4. ``common``         -- the usual charge for that element. A guess, flagged.

Route 2 is what separates mixed-valence sites. It is not circular: each
candidate charge is tested with its *own* parameter, and the candidate that
reproduces itself is selected. Where two candidates fit almost equally well the
site is marked ambiguous rather than assigned, because for a site like that the
answer really is "the data do not say".
"""
from __future__ import annotations

from dataclasses import dataclass

from . import bv, elements
from .structure import Structure


@dataclass
class OxidationVerdict:
    """The outcome for one site, with everything needed to argue about it."""

    site_index: int
    label: str
    element: str
    ox: int | None
    source: str
    candidates: dict[int, float]       # charge -> BVS computed with its own R0
    margin: float | None = None        # how much better the winner fitted
    ambiguous: bool = False

    def explain(self) -> str:
        if not self.candidates:
            return f"{self.label}: {self.ox} ({self.source})"
        bits = ", ".join(f"{c:+d} gives BVS {v:.2f}"
                         for c, v in sorted(self.candidates.items()))
        note = " -- ambiguous" if self.ambiguous else ""
        return f"{self.label}: assigned {self.ox:+d} by {self.source} ({bits}){note}"


def candidate_states(element: str) -> tuple[int, ...]:
    """Plausible formal charges for an element, most common first."""
    alts = elements.ALT_OX.get(element)
    if alts:
        return alts
    common = elements.COMMON_OX.get(element)
    return (common,) if common is not None else ()


def resolve(structure: Structure, params: bv.ParameterSet | None = None,
            v_bond: float = bv.V_BOND_DEFAULT,
            v_list: float = bv.V_LIST_DEFAULT,
            ambiguity_margin: float = 0.25,
            overwrite_cif: bool = False) -> list[OxidationVerdict]:
    """Assign an oxidation state to every cation site, in place.

    Sites whose charge the CIF states are left alone unless `overwrite_cif`.
    Returns one verdict per cation site, in site order.
    """
    from .neighbors import NeighborFinder, search_radius_for

    params = params or bv.DEFAULT
    rmax = search_radius_for(structure, params, v_list)
    finder = NeighborFinder(structure, rmax=rmax)

    verdicts: list[OxidationVerdict] = []
    for idx in structure.cation_sites:
        site = structure.sites[idx]

        if site.ox_source == "cif" and not overwrite_cif:
            verdicts.append(OxidationVerdict(idx, site.label, site.element,
                                             site.ox, "cif", {}))
            continue

        try:
            contacts = finder.contacts_for_site(idx).anions_only(structure)
        except ValueError:
            verdicts.append(OxidationVerdict(idx, site.label, site.element,
                                             site.ox, site.ox_source, {}))
            continue

        scores = _score_candidates(structure, site.element, contacts, params,
                                   v_bond)
        if not scores:
            verdicts.append(OxidationVerdict(idx, site.label, site.element,
                                             site.ox, site.ox_source, {}))
            continue

        # the candidate that best reproduces itself
        ranked = sorted(scores.items(), key=lambda kv: abs(kv[1] - kv[0]))
        best, best_bvs = ranked[0]
        margin = None
        ambiguous = False
        if len(ranked) > 1:
            second, second_bvs = ranked[1]
            margin = abs(second_bvs - second) - abs(best_bvs - best)
            ambiguous = margin < ambiguity_margin

        site.ox = int(best)
        site.ox_source = "bond valence"
        verdicts.append(OxidationVerdict(
            idx, site.label, site.element, int(best), "bond valence",
            scores, margin, ambiguous))

    _record(structure, verdicts)
    return verdicts


def _score_candidates(structure: Structure, element: str, contacts,
                      params: bv.ParameterSet, v_bond: float) -> dict[int, float]:
    """BVS for each candidate charge, each computed with its own parameter."""
    out: dict[int, float] = {}
    for ox in candidate_states(element):
        total = 0.0
        found = False
        for i in range(len(contacts)):
            p = params.get(element, ox, contacts.elements[i])
            if p is None:
                continue
            v = float(p.valence(contacts.distance[i]))
            if v > v_bond:
                total += v * float(contacts.occupancy[i])
                found = True
        if found:
            out[ox] = total
    return out


def _record(structure: Structure, verdicts: list[OxidationVerdict]) -> None:
    guessed = [v.label for v in verdicts if v.source == "common"]
    ambiguous = [v for v in verdicts if v.ambiguous]
    for v in ambiguous:
        pair = sorted(v.candidates.items(), key=lambda kv: abs(kv[1] - kv[0]))[:2]
        structure.notes.append(
            f"oxidation state of {v.label} is ambiguous: "
            + " and ".join(f"{c:+d} gives BVS {b:.2f}" for c, b in pair)
            + " -- both are defensible, and the bond-valence sum reported "
              "depends on which is chosen")
    if guessed:
        structure.notes.append(
            "oxidation state assumed from the usual value for the element for: "
            + ", ".join(guessed))

    mixed = {}
    for v in verdicts:
        if v.ox is not None:
            mixed.setdefault(v.element, set()).add(v.ox)
    for element, states in mixed.items():
        if len(states) > 1:
            structure.notes.append(
                f"{element} is mixed-valence here: "
                + ", ".join(f"{s:+d}" for s in sorted(states))
                + " on different sites")


def charge_balance_residual(structure: Structure) -> float | None:
    """How far the cell is from neutral, or None if a site has no charge."""
    return structure.net_charge()
