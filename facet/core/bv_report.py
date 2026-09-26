"""Bond-valence reports for a whole structure.

The pieces of bond-valence analysis that are about the structure rather than one
site, and that belong on a page you can hand to someone.

The first of them is the cutoff table, which is the argument this program exists
to make. A coordination number quoted with a distance cutoff -- "Bi-O to 3.0 A" --
hides the fact that the cutoff was chosen. State the partial valence instead and
the cutoff follows from it: ``d = R0 - b ln(v)``. The table prints, for every
cation-anion pair in the structure, the distance that corresponds to the chosen
valence thresholds. Two pairs with different R0 then get different distances, as
they should, and nobody has to remember which cutoff was used for which bond.

Nothing here says what a coordination number ought to be.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import bv, elements
from .structure import Structure


@dataclass
class PairCutoff:
    """The distance cutoffs one cation-anion pair gets from a valence threshold."""

    cation: str
    cation_ox: int
    anion: str
    r0: float
    b: float
    fitted: bool
    source: str
    d_bond: float                 # distance at which v = v_bond
    d_list: float                 # distance at which v = v_list
    shortest: float | None = None   # shortest such contact in the structure
    n_bonds: int = 0               # contacts above v_bond
    n_listed: int = 0              # contacts above v_list

    @property
    def label(self) -> str:
        sign = "+" if self.cation_ox > 0 else "-"
        return f"{self.cation}{abs(self.cation_ox)}{sign}–{self.anion}"

    @property
    def window(self) -> float:
        """How much distance lies between the two thresholds."""
        return self.d_list - self.d_bond


@dataclass
class CutoffTable:
    v_bond: float
    v_list: float
    b: float
    parameter_set: str
    rows: list[PairCutoff] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_text(self) -> str:
        lines = [
            f"Distance cutoffs implied by the valence thresholds",
            f"  bond   v >= {self.v_bond:.4f} v.u.",
            f"  listed v >= {self.v_list:.4f} v.u.",
            f"  parameters: {self.parameter_set}",
            "",
            f"{'pair':<14}{'R0':>8}{'b':>7}{'d(bond)':>10}{'d(list)':>10}"
            f"{'window':>9}{'shortest':>10}{'bonds':>7}{'listed':>8}  source",
        ]
        for row in self.rows:
            shortest = "" if row.shortest is None else f"{row.shortest:8.4f}"
            lines.append(
                f"{row.label:<14}{row.r0:8.4f}{row.b:7.3f}{row.d_bond:10.4f}"
                f"{row.d_list:10.4f}{row.window:9.4f}{shortest:>10}"
                f"{row.n_bonds:7d}{row.n_listed:8d}  "
                f"{'fitted' if row.fitted else 'estimated'}")
        for note in self.notes:
            lines.append(f"# {note}")
        return "\n".join(lines)

    def as_csv(self) -> str:
        lines = ["# " + note for note in self.notes]
        lines.append(f"# v_bond = {self.v_bond}, v_list = {self.v_list}, "
                     f"parameters = {self.parameter_set}")
        lines.append("cation,cation_ox,anion,R0,b,d_bond,d_list,window,"
                     "shortest_in_structure,n_bonds,n_listed,provenance,source")
        for row in self.rows:
            shortest = "" if row.shortest is None else f"{row.shortest:.6f}"
            lines.append(
                f"{row.cation},{row.cation_ox},{row.anion},{row.r0:.6f},"
                f"{row.b:.4f},{row.d_bond:.6f},{row.d_list:.6f},"
                f"{row.window:.6f},{shortest},{row.n_bonds},{row.n_listed},"
                f"{'fitted' if row.fitted else 'estimated'},"
                f"\"{row.source}\"")
        return "\n".join(lines) + "\n"


def cutoff_table(structure: Structure, params: bv.ParameterSet | None = None,
                 v_bond: float = bv.V_BOND_DEFAULT,
                 v_list: float = bv.V_LIST_DEFAULT,
                 results=None) -> CutoffTable:
    """The distance cutoff each pair in this structure gets from the thresholds.

    ``results`` may be supplied to fill in what was actually found -- the shortest
    contact of each pair and how many cleared each threshold -- which is what
    turns the table from an assertion into a measurement.
    """
    params = params or bv.DEFAULT
    if not (0 < v_bond and 0 < v_list):
        raise ValueError("the valence thresholds must be positive")

    table = CutoffTable(v_bond=float(v_bond), v_list=float(v_list),
                        b=params.b, parameter_set=params.name)

    # every cation-anion element pair the structure actually contains
    wanted: dict[tuple[str, int, str], None] = {}
    for site in structure.sites:
        if site.is_anion:
            continue
        ox = site.ox if site.ox is not None else elements.COMMON_OX.get(
            site.element)
        if ox is None:
            continue
        for other in structure.sites:
            if other.is_anion:
                wanted.setdefault((site.element, int(ox), other.element), None)

    measured = _measure(structure, results, v_bond, v_list) if results else {}

    missing = []
    for cation, ox, anion in sorted(wanted):
        param = params.get(cation, ox, anion)
        if param is None:
            missing.append(f"{cation}{ox}+–{anion}")
            continue
        found = measured.get((cation, int(ox), anion), {})
        table.rows.append(PairCutoff(
            cation=cation, cation_ox=int(ox), anion=anion,
            r0=param.r0, b=param.b, fitted=param.fitted, source=param.source,
            d_bond=bv.cutoff_for_valence(param.r0, v_bond, param.b),
            d_list=bv.cutoff_for_valence(param.r0, v_list, param.b),
            shortest=found.get("shortest"),
            n_bonds=found.get("n_bonds", 0),
            n_listed=found.get("n_listed", 0)))

    table.notes.append(
        "a cutoff distance is d = R0 - b ln(v); it is not chosen, it follows "
        "from the valence threshold and the pair's own R0")
    if any(not row.fitted for row in table.rows):
        table.notes.append(
            "rows marked estimated use the O'Keeffe-Brese electronegativity "
            "expression rather than a fitted R0, and carry its larger "
            "uncertainty")
    if missing:
        table.notes.append(
            "no parameter was available for: " + ", ".join(missing))
    if not results:
        table.notes.append(
            "the shortest-contact and count columns are empty because no "
            "analysis was supplied")
    return table


def _measure(structure: Structure, results, v_bond: float, v_list: float):
    """What each pair actually looks like in this structure.

    Keyed by the cation's oxidation state as well as the two elements, because
    the table has a row per (cation, charge, anion) -- Bi(3+)-O and Bi(5+)-O are
    different rows with different R0 and different cutoffs. Keying the
    measurement by the element pair alone gave both rows the same counts, so
    every column of measured data was doubled in exactly the structures where it
    matters: Bi4O7, Bi2O4, BaBiO3 and Bi2212 are all mixed-valence, and Bi2212's
    bond count came out 529 against an actual 378.

    A contact whose pair has no parameter has no valence either, and is counted
    in neither column -- its distance is still the shortest contact of that pair,
    because a distance is a measurement and does not depend on a parameter.
    """
    out: dict[tuple[str, int, str], dict] = {}
    for result in results:
        site = structure.sites[result.site_index]
        ox = site.ox if site.ox is not None else elements.COMMON_OX.get(
            site.element)
        if ox is None:
            continue
        for contact in getattr(result, "contacts", []):
            anion = getattr(contact, "element", None)
            if anion is None:
                continue
            key = (site.element, int(ox), elements.normalise(anion))
            entry = out.setdefault(key, {"shortest": None, "n_bonds": 0,
                                         "n_listed": 0})
            distance = float(contact.distance)
            if entry["shortest"] is None or distance < entry["shortest"]:
                entry["shortest"] = distance
            if contact.valence is None:
                continue
            valence = float(contact.valence)
            if valence >= v_list:
                entry["n_listed"] += 1
            if valence >= v_bond:
                entry["n_bonds"] += 1
    return out


# ---------------------------------------------------------------------------
# threshold sensitivity
# ---------------------------------------------------------------------------

@dataclass
class ThresholdScan:
    """How every site's coordination number responds to the threshold.

    The thing a distance cutoff hides. A site whose CN is the same across a wide
    range of thresholds has a coordination number worth quoting; one that changes
    with every step does not, and no choice of cutoff makes it so.
    """

    thresholds: np.ndarray
    per_site: dict[str, np.ndarray] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def stable_range(self, label: str, at: float) -> tuple[float, float] | None:
        """The span of thresholds over which this site keeps the CN it has at
        ``at``. None if the label is unknown."""
        counts = self.per_site.get(label)
        if counts is None:
            return None
        index = int(np.argmin(np.abs(self.thresholds - at)))
        target = counts[index]
        low = index
        while low > 0 and counts[low - 1] == target:
            low -= 1
        high = index
        while high < len(counts) - 1 and counts[high + 1] == target:
            high += 1
        return float(self.thresholds[low]), float(self.thresholds[high])

    def widths(self, at: float) -> dict[str, float]:
        """How wide each site's stable range is at this threshold, in v.u."""
        out = {}
        for label in self.per_site:
            span = self.stable_range(label, at)
            if span is not None:
                out[label] = span[1] - span[0]
        return out

    def as_csv(self) -> str:
        lines = ["# " + note for note in self.notes]
        labels = sorted(self.per_site)
        lines.append("v_bond," + ",".join(labels))
        for i, threshold in enumerate(self.thresholds):
            row = [f"{threshold:.6f}"]
            row.extend(str(int(self.per_site[label][i])) for label in labels)
            lines.append(",".join(row))
        return "\n".join(lines) + "\n"


def threshold_scan(results, low: float = 0.005, high: float = 0.30,
                   steps: int = 120) -> ThresholdScan:
    """Coordination number against the valence threshold, for every site.

    Computed from contacts that were already found, so it costs no neighbour
    search: a contact's valence does not depend on the threshold, only whether it
    is counted does. That is the same property that makes the cutoff slider free.
    """
    thresholds = np.linspace(float(low), float(high), int(steps))
    scan = ThresholdScan(thresholds=thresholds)
    for result in results:
        valences = np.array([float(c.valence)
                             for c in getattr(result, "contacts", [])
                             if c.valence is not None], float)
        if valences.size:
            counts = (valences[None, :] >= thresholds[:, None]).sum(axis=1)
        else:
            counts = np.zeros(len(thresholds), int)
        scan.per_site[result.label] = counts.astype(int)
    scan.notes.append(
        "coordination number as a function of the partial-valence threshold; "
        "a wide plateau means the count does not depend on where the line is "
        "drawn, a narrow one means it does")
    scan.notes.append(
        "computed from contacts already found, so no neighbour search was "
        "repeated: whether a contact is counted depends on the threshold, its "
        "valence does not")
    return scan


# ---------------------------------------------------------------------------
# anions
# ---------------------------------------------------------------------------

@dataclass
class AnionSum:
    """A bond-valence sum for an anion, from the cations around it.

    Two sums, matching the two the cation side reports: ``bvs`` over the contacts
    above the bond threshold and ``bvs_listed`` over everything above the listing
    threshold. Comparing the wrong one against a cation total makes the charge
    balance disagree by the weight of the contacts between the two thresholds --
    which for alpha-Bi2O3 is about a percent, small enough to be mistaken for
    rounding.
    """

    label: str
    element: str
    expected: float | None
    bvs: float
    n_contacts: int
    discrepancy: float | None
    bvs_listed: float = float("nan")
    n_listed: int = 0

    @property
    def is_balanced(self) -> bool:
        """Whether the sum matches the formal charge to within 0.2 v.u.

        A threshold, stated as one, not a verdict about the structure: 0.2 is
        conventional and nothing more.
        """
        return self.discrepancy is not None and abs(self.discrepancy) <= 0.2


def anion_sums(structure: Structure, params: bv.ParameterSet | None = None,
               v_bond: float = bv.V_BOND_DEFAULT,
               v_list: float = bv.V_LIST_DEFAULT) -> list[AnionSum]:
    """Bond-valence sums for the anions, from a search around each anion.

    Usually only the cations are reported, and an anion's sum is the other half
    of the same check. It catches things the cation sums do not: an oxygen whose
    sum comes to 1.4 is either missing a bond to something the file left out,
    such as a hydrogen, or is not where the refinement put it.

    This does its own neighbour search rather than re-gathering the cation
    results, and the reason is worth stating because the shortcut is tempting and
    wrong. A cation site's contact list belongs to one representative atom of
    that site. An anion is reached by cation atoms from every equivalent position,
    most of which are not that representative -- so summing the cation lists gives
    each anion only the fraction of its bonds that happen to touch the
    representatives. On alpha-Bi2O3 that came to 0.66 v.u. for an O(2-): a number
    low enough to look like a finding rather than an error.
    """
    from .neighbors import NeighborFinder, search_radius_for

    params = params or bv.DEFAULT
    anion_indices = [i for i, site in enumerate(structure.sites)
                     if site.is_anion]
    if not anion_indices or not structure.atoms:
        # Nothing to search around, or nothing to search. A structure whose
        # sites were never expanded is a legitimate thing to be handed -- a
        # report should come back empty rather than raise from three layers down.
        return []

    rmax = search_radius_for(structure, params, v_list)
    finder = NeighborFinder(structure, rmax=rmax)

    out = []
    for index in anion_indices:
        site = structure.sites[index]
        contacts = finder.contacts_for_site(index)
        bonded = listed = 0.0
        n_bonded = n_listed = 0
        for i in range(len(contacts)):
            neighbour_site = structure.sites[contacts.neighbor_site[i]]
            if neighbour_site.is_anion:
                continue          # an anion-anion contact is not a bond here
            param = params.get(neighbour_site.element, neighbour_site.ox,
                               site.element)
            if param is None:
                continue
            valence = float(param.valence(contacts.distance[i]))
            if valence < v_list:
                continue
            # weighted by the *cation's* occupancy: a half-occupied cation site
            # delivers half the valence to the anion it reaches
            weighted = valence * float(contacts.occupancy[i])
            listed += weighted
            n_listed += 1
            if valence >= v_bond:
                bonded += weighted
                n_bonded += 1

        expected = abs(site.ox) if site.ox is not None else None
        out.append(AnionSum(
            label=site.label, element=site.element, expected=expected,
            bvs=bonded, n_contacts=n_bonded,
            discrepancy=(bonded - expected) if expected is not None else None,
            bvs_listed=listed, n_listed=n_listed))
    return out


def charge_balance(structure: Structure, results,
                   params: bv.ParameterSet | None = None,
                   v_bond: float = bv.V_BOND_DEFAULT,
                   v_list: float = bv.V_LIST_DEFAULT) -> dict:
    """Do the cation and anion valence sums account for each other?

    Three numbers: the total valence the cations distribute over the cell, the
    total the anions receive, and the difference. They are the same sum counted
    from either end, so they must agree -- and because the two are computed by
    separate searches, from opposite directions, agreement is a real check on
    both. A difference means a contact was counted on one side and not the other,
    which is a fact about the analysis rather than about the structure.

    Each sum is per unit cell: a site's value times its multiplicity, times its
    own occupancy on the side where that occupancy has not already been applied.
    """
    params = params or bv.DEFAULT

    cation_total = cation_listed = 0.0
    excluded = []
    for result in results:
        site = structure.sites[result.site_index]
        if site.is_anion:
            continue
        value = float(result.bvs)
        if value != value:          # nan: no parameter covered this site's pairs
            excluded.append(site.label)
            continue
        multiplicity = site.multiplicity or 1
        # The cation's BVS already carries each anion's occupancy through its
        # contacts, so only the cation's own occupancy is applied here.
        weight = multiplicity * site.occupancy
        cation_total += value * weight
        listed = float(getattr(result, "bvs_listed", float("nan")))
        cation_listed += (listed if listed == listed else value) * weight

    anion_total = anion_listed = 0.0
    for row in anion_sums(structure, params, v_bond, v_list):
        index = structure.site_by_label(row.label)
        site = structure.sites[index] if index is not None else None
        multiplicity = (site.multiplicity or 1) if site else 1
        occupancy = site.occupancy if site else 1.0
        # and symmetrically: the anion's sum carries the cations' occupancies,
        # so only the anion's own is applied
        weight = multiplicity * occupancy
        anion_total += row.bvs * weight
        anion_listed += (row.bvs_listed if row.bvs_listed == row.bvs_listed
                         else row.bvs) * weight

    difference = cation_total - anion_total
    scale = max(abs(cation_total), abs(anion_total), 1e-12)
    return {
        "cation_valence": cation_total,
        "anion_valence": anion_total,
        "difference": difference,
        "relative_difference": difference / scale,
        "cation_valence_listed": cation_listed,
        "anion_valence_listed": anion_listed,
        "difference_listed": cation_listed - anion_listed,
        "excluded_sites": tuple(excluded),
        "note": _balance_note(difference, scale, excluded),
    }


def _balance_note(difference: float, scale: float, excluded) -> str:
    """What a non-zero difference actually tells you.

    The two totals are the same bonds counted from opposite ends by two separate
    neighbour searches, so they agree exactly when a structure satisfies its own
    symmetry -- which most do, to machine precision. When they do not, the
    difference measures how far the deposited coordinates fall short of the space
    group they declare: in the P3 BiI3 file, bonds that symmetry requires to be
    equal differ by 0.0006 A, and the sums differ by 0.05 per cent. That is
    worth knowing and is a fact about the file, so it is reported rather than
    rounded away.
    """
    relative = abs(difference) / scale
    lines = ["the two totals are the same bonds counted from opposite ends by "
             "separate searches"]
    if relative < 1e-9:
        lines.append("they agree exactly, so the coordinates satisfy the "
                     "declared symmetry to machine precision")
    else:
        lines.append(
            f"they differ by {relative * 100:.3f} per cent, which measures how "
            "far the coordinates fall short of the symmetry the file declares: "
            "bonds that should be equivalent are not quite equal")
    if excluded:
        lines.append(
            "no parameter covered " + ", ".join(excluded)
            + ", so those sites are left out of both totals")
    return "; ".join(lines)
