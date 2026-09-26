"""Structure health: what is wrong with this file, stated as measurements.

Real collections contain broken entries. The reference set used to build FACET
holds an atom loop of ``? ? ? ?``, oxygen on the wrong Wyckoff site, a P–O
distance of 1.10 Å in a diffraction-card export, and coordinate esds of 0.07 in
fractional units. Reading any of those as if it were sound produces a
confident, wrong number.

Each check returns what it measured and the value it compared against. None of
them says whether a structure is usable — that depends on the question, and a
file that is useless for a bond-valence sum may be perfectly good for indexing
a powder pattern.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from . import elements


class Level(IntEnum):
    """How far a measurement is from the ordinary range.

    Not a verdict. NOTE means "worth knowing", CHECK means "outside what is
    usually seen", IMPOSSIBLE means "no structure has ever shown this".
    """

    NOTE = 0
    CHECK = 1
    IMPOSSIBLE = 2


@dataclass
class Finding:
    level: Level
    code: str
    message: str
    where: str = ""
    value: float | None = None
    reference: float | None = None

    def __str__(self) -> str:
        return f"[{self.level.name}] {self.where + ': ' if self.where else ''}{self.message}"


@dataclass
class HealthReport:
    findings: list[Finding] = field(default_factory=list)

    def collapse(self) -> None:
        """Fold findings that say the same thing into one, with a count.

        A close pair of atoms is found once per symmetry copy, so a single
        shared site in a cubic structure produced sixteen identical lines. That
        is not more information than one line -- it is the same information made
        hard to read, and a report nobody reads is a report that does not work.
        The count is kept, because how many copies there are is worth knowing.
        """
        seen: dict[tuple, Finding] = {}
        counts: dict[tuple, int] = {}
        order: list[tuple] = []
        for finding in self.findings:
            key = (finding.level, finding.code, finding.message)
            if key not in seen:
                seen[key] = finding
                counts[key] = 1
                order.append(key)
            else:
                counts[key] += 1
                # keep the first `where`, but remember there were others
                if finding.where and finding.where not in seen[key].where:
                    if seen[key].where.count(",") < 2:
                        seen[key].where = (seen[key].where + ", "
                                           + finding.where).strip(", ")

        collapsed = []
        for key in order:
            finding = seen[key]
            count = counts[key]
            if count > 1:
                finding.message = (
                    f"{finding.message} \u2014 {count} symmetry-equivalent "
                    "occurrences")
            collapsed.append(finding)
        self.findings = collapsed

    @property
    def worst(self) -> Level | None:
        return max((f.level for f in self.findings), default=None)

    def of_level(self, level: Level) -> list[Finding]:
        return [f for f in self.findings if f.level == level]

    def __bool__(self) -> bool:
        return bool(self.findings)

    def __len__(self) -> int:
        return len(self.findings)


# Shortest bond ever observed between two non-metal atoms is about 1.09 A
# (H-H in dihydrogen is 0.74, but that is not a case FACET meets). Below 1.2 A
# between heavy atoms, the coordinates are wrong.
MIN_PHYSICAL_BOND = 1.20
# Below this, two positions are a split site rather than two atoms.
SPLIT_POSITION = 0.80


def check(structure, results=None, v_bond: float = 0.075) -> HealthReport:
    """Every check, over one structure and optionally its analysis."""
    report = HealthReport()
    _check_coordinates(structure, report)
    _check_contacts(structure, report)
    _check_occupancy(structure, report)
    _check_charge(structure, report)
    _check_esds(structure, report)
    _check_formula(structure, report)
    if results:
        _check_valences(results, report, v_bond)
    for note in structure.notes:
        report.findings.append(Finding(Level.NOTE, "note", note))
    report.collapse()
    return report


def parse_formula(text) -> dict[str, float]:
    """A CIF ``_chemical_formula_sum`` as element counts.

    Handles the forms these files use: ``Bi2 O3``, ``Bi0.92 O1.54 Si0.08``,
    ``Na3 Bi (P O4)2`` -- the last by expanding the bracketed group, because a
    formula that cannot be read is not a formula that can be checked against.
    """
    import re

    if not text:
        return {}
    text = str(text).strip().strip("'\"")
    if not text:
        return {}

    def accumulate(body: str, factor: float, into: dict) -> None:
        for symbol, count in re.findall(r"([A-Z][a-z]?)\s*([0-9]*\.?[0-9]*)",
                                       body):
            if not symbol:
                continue
            amount = float(count) if count else 1.0
            into[symbol] = into.get(symbol, 0.0) + amount * factor

    out: dict[str, float] = {}
    # bracketed groups first, then whatever is left
    remainder = text
    for group, multiplier in re.findall(r"\(([^()]*)\)\s*([0-9]*\.?[0-9]*)",
                                        text):
        accumulate(group, float(multiplier) if multiplier else 1.0, out)
        remainder = remainder.replace(f"({group}){multiplier}", " ", 1)
        remainder = remainder.replace(f"({group})", " ", 1)
    accumulate(remainder, 1.0, out)
    return {k: v for k, v in out.items() if v > 0}


def _check_formula(structure, report: HealthReport) -> None:
    """Does the expanded cell contain what the file says it contains?

    This is the check that catches a symmetry expansion going wrong, and it
    caught one: an Fm-3m entry whose oxygen sits on the 32-fold position came out
    with 8 oxygen atoms instead of 32, so the cell held a quarter of the oxygen
    the formula states -- and therefore a quarter of the anion charge, with every
    bond-valence sum and every diffraction intensity wrong in consequence. None
    of the other checks noticed, because the structure was internally consistent.
    The formula is the one statement in the file that is independent of the
    coordinates and the symmetry, which is exactly what makes it useful here.

    Compared as ratios, not absolute counts, because the formula is per formula
    unit and the cell holds Z of them -- and Z is often absent or wrong.
    """
    declared = parse_formula(getattr(structure, "formula", None))
    if not declared:
        return
    found = structure.composition()
    if not found:
        return

    shared = sorted(set(declared) & set(found))
    if len(shared) < 2:
        # with one element in common there is no ratio to compare
        missing = sorted(set(declared) - set(found))
        if missing:
            report.findings.append(Finding(
                Level.CHECK, "formula-element",
                f"the formula names {', '.join(missing)} but the expanded cell "
                "contains none"))
        return

    reference = shared[0]
    worst = (0.0, "")
    for element in shared[1:]:
        want = declared[element] / declared[reference]
        got = found[element] / found[reference]
        if want <= 0:
            continue
        error = abs(got - want) / want
        if error > worst[0]:
            worst = (error, f"{element}:{reference} is {got:.4g} where the "
                            f"formula gives {want:.4g}")
    # 2 per cent absorbs rounded occupancies and a formula quoted to two figures
    if worst[0] > 0.02:
        report.findings.append(Finding(
            Level.CHECK, "formula-mismatch",
            f"the expanded cell does not match the stated formula "
            f"{structure.formula!r}: {worst[1]}",
            value=worst[0]))

    extra = sorted(set(found) - set(declared))
    if extra:
        report.findings.append(Finding(
            Level.NOTE, "formula-extra",
            f"the cell contains {', '.join(extra)}, which the stated formula "
            f"{structure.formula!r} does not name"))


def _check_coordinates(structure, report: HealthReport) -> None:
    if not structure.atoms:
        report.findings.append(Finding(
            Level.IMPOSSIBLE, "no-atoms",
            "the file declares a cell but no atomic coordinates"))
        return
    for atom in structure.atoms:
        if not np.all(np.isfinite(atom.frac)):
            report.findings.append(Finding(
                Level.IMPOSSIBLE, "bad-coordinate",
                "coordinates are not finite", atom.label))


def _check_contacts(structure, report: HealthReport) -> None:
    """Distances no real structure shows.

    Split positions are reported separately from impossible contacts: a pair
    0.3 A apart is a disorder model, a pair 1.1 A apart is an error.
    """
    if len(structure.atoms) < 2:
        return
    from scipy.spatial import cKDTree

    positions = structure.cart_array()
    tree = cKDTree(positions)
    for i, j in tree.query_pairs(MIN_PHYSICAL_BOND):
        d = float(np.linalg.norm(positions[i] - positions[j]))
        a, b = structure.atoms[i], structure.atoms[j]
        occ = min(a.occupancy, b.occupancy)
        if d < SPLIT_POSITION or occ < 0.99:
            report.findings.append(Finding(
                Level.NOTE, "split-position",
                f"{a.label} and {b.label} are {d:.3f} Å apart with occupancy "
                f"{a.occupancy:.2f} and {b.occupancy:.2f} — a split or shared "
                f"site", f"{a.label}/{b.label}", d, SPLIT_POSITION))
        else:
            report.findings.append(Finding(
                Level.IMPOSSIBLE, "short-contact",
                f"{a.label}–{b.label} is {d:.3f} Å, shorter than any "
                f"interatomic distance ({MIN_PHYSICAL_BOND:.2f} Å)",
                f"{a.label}/{b.label}", d, MIN_PHYSICAL_BOND))


def _check_occupancy(structure, report: HealthReport) -> None:
    for site in structure.sites:
        if site.occupancy > 1.0001:
            report.findings.append(Finding(
                Level.IMPOSSIBLE, "occupancy-above-one",
                f"occupancy {site.occupancy:.3f}", site.label,
                site.occupancy, 1.0))
        elif site.occupancy <= 0.0:
            report.findings.append(Finding(
                Level.CHECK, "occupancy-zero",
                f"occupancy {site.occupancy:.3f}", site.label,
                site.occupancy, 0.0))


def _check_charge(structure, report: HealthReport) -> None:
    charge = structure.net_charge()
    if charge is None:
        return
    total = sum(abs(s.ox or 0) * s.occupancy * (s.multiplicity or 1)
                for s in structure.sites) or 1.0
    if abs(charge) > 0.02 * total:
        report.findings.append(Finding(
            Level.CHECK, "charge-imbalance",
            f"the cell carries a net charge of {charge:+.3f} against "
            f"{total:.1f} of formal charge",
            value=charge, reference=0.0))


def _check_esds(structure, report: HealthReport) -> None:
    """Coordinate uncertainties large enough to swallow a bond length.

    An esd of 0.01 in fractional units on a 10 Å axis is 0.1 Å, which is the
    size of the differences a coordination analysis turns on.
    """
    for site in structure.sites:
        if site.frac_esd is None:
            continue
        lengths = np.array(structure.cell.lengths, float)
        absolute = np.asarray(site.frac_esd, float) * lengths
        worst = float(np.max(absolute))
        if worst > 0.10:
            report.findings.append(Finding(
                Level.CHECK, "large-esd",
                f"coordinate uncertainty reaches {worst:.3f} Å", site.label,
                worst, 0.10))
        elif worst > 0.03:
            report.findings.append(Finding(
                Level.NOTE, "esd",
                f"coordinate uncertainty reaches {worst:.3f} Å", site.label,
                worst, 0.03))


def _check_valences(results, report: HealthReport, v_bond: float) -> None:
    """Bond-valence sums far from the assumed oxidation state.

    The R0 values themselves carry about ±0.02 Å, which is roughly ±6 % in a
    sum, so a few per cent means nothing. A sum off by half its own value means
    the coordinates, the oxidation state or the parameter is wrong, and the
    check does not say which.
    """
    for r in results:
        if r.ox is None or not r.bonds:
            continue
        expected = abs(r.ox)
        if expected == 0:
            continue
        ratio = r.bvs / expected
        if ratio > 1.5 or ratio < 0.5:
            report.findings.append(Finding(
                Level.IMPOSSIBLE, "valence-far-off",
                f"bond-valence sum {r.bvs:.2f} v.u. against an assumed "
                f"{r.ox:+d} ({100 * (ratio - 1):+.0f} %)",
                r.label, r.bvs, float(expected)))
        elif ratio > 1.25 or ratio < 0.75:
            report.findings.append(Finding(
                Level.CHECK, "valence-off",
                f"bond-valence sum {r.bvs:.2f} v.u. against an assumed "
                f"{r.ox:+d} ({100 * (ratio - 1):+.0f} %)",
                r.label, r.bvs, float(expected)))


def summarise(report: HealthReport) -> str:
    """One line per finding, worst first."""
    if not report:
        return "no findings"
    order = sorted(report.findings, key=lambda f: -int(f.level))
    return "\n".join(str(f) for f in order)
